from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, List
import copy, math
import torch


@dataclass
class ReduceOnPlateauMAP50_WithDetectorNNClone:
    """External AP50 based learning rate scheduler for Ultralytics YOLO.

    This callback evaluates AP at IoU 0.50 (AP50) using an external detector
    clone and a COCO style evaluation function, and reduces the learning rate
    when performance stalls.

    Workflow at selected epochs:

    1. Copy the current network weights from ``owner.model`` into a new detector
       instance created by ``detector_factory``.
    2. Run ``clone_det.predict_sticker(**predict_kwargs)`` on a chosen split
       (usually the train split).
    3. Compute AP50 with ``evaluate_fn`` on a COCO split at ``coco_split_dir``.
    4. If AP50 stops improving, apply a learning rate reduction on the original
       optimizer inside the YOLO trainer.

    The callback can also perform periodic evaluations on a separate test split
    without affecting the learning rate ("test_eval" options).

    This class is designed to be registered as a Ultralytics YOLO callback:

    .. code-block:: python

        model.add_callback("on_train_epoch_start", plateau_cb.on_train_epoch_start)
        model.add_callback("on_fit_epoch_end", plateau_cb)
        model.add_callback("on_fit_end", plateau_cb.on_fit_end)
        model.add_callback("on_train_end", plateau_cb.on_train_end)

    Attributes:
        factor: Multiplicative factor for reducing the learning rate when a
            plateau is detected.
        patience: Number of evaluation steps to wait before reducing the
            learning rate the first time.
        patience_after_first: Optional different patience value after the first
            reduction. If ``None``, reuse ``patience``.
        cooldown: Number of evaluation steps to wait after a reduction before
            allowing another.
        min_lr: Lower bound for the learning rate.

        warmup_epochs: Number of epochs to skip before activating plateau logic.
        start_after_map: Minimum AP50 required before plateau detection can
            start (for example ignore very early noisy epochs).

        use_plateau: If ``False``, only scheduled learning rate changes are
            applied and plateau detection is disabled.

        scheduled_epochs: Epoch indices where a scheduled LR change should
            occur.
        scheduled_factors: Optional per epoch factors to use at scheduled
            epochs. If missing, ``factor`` is used.
        scheduled_set_lrs: Optional mapping from epoch index to an absolute
            learning rate value to set at that epoch.

        owner: Owning detector wrapper (for example YOLODetector) that holds
            the training model as ``owner.model``.
        detector_factory: Callable that returns a new detector instance of the
            same type as ``owner``. Used to host the cloned network.
        evaluate_fn: Callable that computes AP50 given a COCO split and
            predictions. The expected signature is roughly:

                evaluate_fn(coco_split_dir, predictions, yolo_names=None, ...)

        coco_split_dir: Path to the main COCO split for AP50 evaluation.
        yolo_names: Optional list of class names for mapping YOLO class ids to
            COCO category ids.
        predict_kwargs: Keyword arguments passed to
            ``clone_det.predict_sticker`` for the main evaluation.
        eval_every: Evaluate and possibly adjust LR every N epochs.
        clone_device: Device where the cloned network is placed ("cpu" or
            "cuda:0").
        verbose: If ``True``, print evaluation and LR change logs.

        test_eval_every: Frequency for periodic test evaluations. Set to
            ``None`` or ``0`` to disable.
        test_eval_start_epoch: Epoch index at which to start test evaluations.
        test_eval_epochs: Explicit list of epochs which should always be
            evaluated on the test split, in addition to the frequency rule.
        test_predict_kwargs: Extra kwargs for test predictions.
        test_coco_split_dir: Optional separate COCO split for test evaluation.
        test_ap_history: Recorded history of test AP50 values.

        history: List of dictionaries containing epoch, train loss, AP50, and
            learning rates for each evaluation step.
    """

    # Scheduling
    factor: float = 0.5
    patience: int = 10
    patience_after_first: Optional[int] = None
    cooldown: int = 0
    min_lr: float = 1e-6

    # Activation gates
    warmup_epochs: int = 0
    start_after_map: float = 0.0

    # Plateau on/off
    use_plateau: bool = True

    # Scheduled LR changes
    scheduled_epochs: List[int] = field(default_factory=list)
    scheduled_factors: Dict[int, float] = field(default_factory=dict)
    scheduled_set_lrs: Dict[int, float] = field(default_factory=dict)

    # Wiring
    owner: Any = None                                   # YOLODetector (has .model and .predict_sticker)
    detector_factory: Optional[Callable[[], Any]] = None# returns NEW YOLODetector
    evaluate_fn: Optional[Callable[..., float]] = None  # returns AP50
    coco_split_dir: str = ""                            # GT dir for main eval
    yolo_names: Optional[List[str]] = None
    predict_kwargs: Optional[Dict[str, Any]] = None     # main eval (usually split="train")
    eval_every: int = 1
    clone_device: Optional[str] = "cpu"
    verbose: bool = True

    # ---- Periodic TEST evaluation ----
    test_eval_every: Optional[int] = 25                 # frequency after start; 0 or None disables
    test_eval_start_epoch: int = 0                      # start checking at this epoch (inclusive)
    test_eval_epochs: Optional[List[int]] = None        # explicit extra epochs to evaluate (in addition to rule)
    test_predict_kwargs: Optional[Dict[str, Any]] = None
    test_coco_split_dir: Optional[str] = None
    test_ap_history: List[Dict[str, Any]] = field(default_factory=list)

    # State
    _best: float = float("-inf")
    _bad: int = 0
    _cool: int = 0
    _active: bool = False

    # Logging
    history: List[Dict[str, Any]] = field(default_factory=list)
    _reduced_flag: bool = False
    _lr_after: Optional[List[float]] = None
    _printed: bool = False
    _target_lrs: Optional[List[float]] = None
    _had_reduction: bool = False

    # Track which scheduled steps already applied
    _scheduled_applied: set = field(default_factory=set)

    def __call__(self, trainer):
        """Main callback body run at the end of a training epoch.

        This method:

          1. Clones the current detector network.
          2. Evaluates AP50 on the main split.
          3. Performs scheduled learning rate changes.
          4. Optionally applies plateau based learning rate reductions.
          5. Optionally evaluates AP50 on a test split.
          6. Logs the results into the history list.

        It is meant to be registered as the handler for the Ultralytics
        ``on_fit_epoch_end`` event.
        """
        epoch = int(getattr(trainer, "epoch", 0))
        if epoch < self.warmup_epochs:
            return
        if self.eval_every > 1 and (epoch % self.eval_every):
            return
        if not (self.owner and self.detector_factory and self.evaluate_fn):
            return

        # Reset per epoch flags
        self._reduced_flag = False
        self._lr_after = None

        # Training loss snapshot
        train_loss = self._get_train_loss(trainer)

        # 1) Grab training nn.Module safely
        ultra_train = getattr(self.owner, "model", None)
        if ultra_train is None:
            return
        nn_train = getattr(ultra_train, "model", ultra_train)

        # 2) Fresh YOLODetector clone and deepcopy nn.Module
        clone_det = self.detector_factory()
        ultra_clone = getattr(clone_det, "model", None)
        if ultra_clone is None:
            return

        nn_copied = copy.deepcopy(nn_train)
        nn_copied.eval()
        if self.clone_device:
            nn_copied.to(self.clone_device)
        for p in nn_copied.parameters():
            p.requires_grad_(False)

        if hasattr(ultra_clone, "model"):
            ultra_clone.model = nn_copied
        else:
            clone_det.model = nn_copied

        # carry metadata if present
        for attr in ("names", "nc", "args"):
            if hasattr(ultra_train, attr) and hasattr(ultra_clone, attr):
                try:
                    setattr(ultra_clone, attr, copy.deepcopy(getattr(ultra_train, attr)))
                except Exception:
                    pass

        # 3) Predict with the clone, then evaluate AP50 on main split
        with torch.inference_mode():
            preds = clone_det.predict_sticker(**(self.predict_kwargs or {}))

        m50 = float(
            self.evaluate_fn(
                coco_split_dir=self.coco_split_dir,
                predictions=preds,
                yolo_names=self.yolo_names,
            )
            or 0.0
        )

        if self.verbose:
            print(f"[ExtEval:det-nn-clone] epoch {epoch} mAP50(main)={m50:.6f}")

        # 3.5) Scheduled LR events
        self._maybe_scheduled_step(trainer, epoch)

        # 4) Plateau logic on original optimizer
        if self.use_plateau:
            if not self._active:
                if m50 > self.start_after_map:
                    self._active = True
                    self._best = m50
                    self._bad = 0
                    if self.verbose:
                        print(f"[Plateau] activated at epoch {epoch}: mAP50={m50:.6f}")
            else:
                if m50 > self._best + 1e-12:
                    self._best = m50
                    self._bad = 0
                    if self.verbose:
                        print(f"[Plateau] new best mAP50={m50:.6f} (epoch {epoch})")
                else:
                    self._bad += 1
                    if self._cool > 0:
                        self._cool -= 1
                    else:
                        eff_patience = (
                            self.patience
                            if not self._had_reduction
                            else (self.patience_after_first or self.patience)
                        )
                        if self._bad >= eff_patience:
                            self._reduce_lr(trainer)  # sets self._target_lrs
                            self._bad = 0
                            self._cool = self.cooldown

        # 5) Periodic test evaluation (does not change LR)
        do_test = False
        if self.test_eval_every and self.test_eval_every > 0:
            start = int(self.test_eval_start_epoch or 0)
            if epoch >= start and ((epoch - start) % self.test_eval_every == 0):
                do_test = True
        if self.test_eval_epochs and epoch in self.test_eval_epochs:
            do_test = True

        if do_test:
            tk = dict(self.predict_kwargs or {})
            tk["split"] = "test"
            if self.test_predict_kwargs:
                tk.update(self.test_predict_kwargs)

            with torch.inference_mode():
                preds_test = clone_det.predict_sticker(**tk)

            test_dir = self.test_coco_split_dir or self.coco_split_dir
            m50_test = float(
                self.evaluate_fn(
                    coco_split_dir=test_dir,
                    predictions=preds_test,
                    yolo_names=self.yolo_names,
                )
                or 0.0
            )
            self.test_ap_history.append({"epoch": epoch, "ap50": m50_test})
            if self.verbose:
                print(f"[ExtEval:TEST] epoch {epoch} mAP50(test)={m50_test:.6f}")

        # 6) Log after any LR change
        cur_lrs = self._lr_after if self._lr_after is not None else self._get_lrs(trainer)
        self.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "ap50": m50,
                "lrs": cur_lrs,
                "reduced": self._reduced_flag,
            }
        )

        # If last epoch, print now
        total_epochs = self._get_total_epochs(trainer)
        if total_epochs is not None and (epoch + 1) >= int(total_epochs) and not self._printed:
            self._print_history()
            self._printed = True

        # Keep target LRs in sync even if no reduction occurred
        if cur_lrs is not None:
            self._target_lrs = list(cur_lrs)

    # ---------- enforce our LR at start of each epoch ----------
    def on_train_epoch_start(self, trainer):
        """Callback hook to enforce the current target learning rates.

        This should be registered on the Ultralytics event
        ``on_train_epoch_start`` so that any changes to learning rates are
        applied consistently at the beginning of each epoch.
        """
        self._apply_target_lrs(trainer)

    def _apply_target_lrs(self, trainer):
        if not self._target_lrs:
            return
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        for pg, v in zip(opt.param_groups, self._target_lrs):
            pg["lr"] = float(v)
            if "initial_lr" in pg:
                pg["initial_lr"] = float(v)
        args = getattr(trainer, "args", None)
        try:
            if args is not None and hasattr(args, "lr0"):
                setattr(args, "lr0", float(min(self._target_lrs)))
        except Exception:
            pass
        try:
            if hasattr(trainer, "lr0"):
                if isinstance(trainer.lr0, (list, tuple)):
                    trainer.lr0 = list(self._target_lrs)
                else:
                    trainer.lr0 = float(min(self._target_lrs))
        except Exception:
            pass
        sch = getattr(trainer, "scheduler", None)
        if sch is not None and hasattr(sch, "base_lrs"):
            try:
                sch.base_lrs = [float(v) for v in self._target_lrs]
            except Exception:
                pass

    # ---- Scheduled steps ----
    def _maybe_scheduled_step(self, trainer, epoch: int):
        if epoch in self._scheduled_applied:
            return
        if epoch in self.scheduled_set_lrs:
            target = float(self.scheduled_set_lrs[epoch])
            self._set_absolute_lr(trainer, target)
            self._scheduled_applied.add(epoch)
            if self.verbose:
                print(f"[Schedule] epoch {epoch}: set LR -> {target}")
            return
        if (epoch in self.scheduled_epochs) or (epoch in self.scheduled_factors):
            fac = float(self.scheduled_factors.get(epoch, self.factor))
            self._reduce_lr(trainer, factor=fac)
            self._scheduled_applied.add(epoch)
            if self.verbose:
                print(f"[Schedule] epoch {epoch}: reduce LR by factor {fac}")

    def _set_absolute_lr(self, trainer, target_lr: float):
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        new_vals = []
        for pg in opt.param_groups:
            pg["lr"] = max(float(target_lr), self.min_lr)
            new_vals.append(float(pg["lr"]))
        self._reduced_flag = True
        self._lr_after = new_vals
        self._target_lrs = list(new_vals)
        self._had_reduction = True
        self._apply_target_lrs(trainer)

    def _reduce_lr(self, trainer, factor: Optional[float] = None):
        if not hasattr(trainer, "optimizer") or trainer.optimizer is None:
            return
        fac = float(factor if factor is not None else self.factor)
        new_vals = []
        for pg in trainer.optimizer.param_groups:
            old = float(pg.get("lr", 0.0))
            new = max(old * fac, self.min_lr)
            if new < old - 1e-12:
                pg["lr"] = new
            new_vals.append(pg.get("lr", old))
        self._reduced_flag = True
        self._lr_after = new_vals
        self._target_lrs = list(new_vals)
        if not self._had_reduction:
            self._had_reduction = True
            if self.patience_after_first is not None:
                self.patience = self.patience_after_first
        self._apply_target_lrs(trainer)
        if self.verbose:
            print(f"[Plateau] LR reduced -> {new_vals}")

    # Helpers
    def _get_lrs(self, trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return None
        try:
            return [float(pg.get("lr", 0.0)) for pg in opt.param_groups]
        except Exception:
            return None

    def _get_train_loss(self, trainer) -> Optional[float]:
        """Try to extract the current training loss from the trainer object."""
        for attr in ("tloss", "train_loss", "loss"):
            if hasattr(trainer, attr):
                v = getattr(trainer, attr)
                try:
                    if torch.is_tensor(v):
                        return float(v.detach().cpu().item())
                    return float(v)
                except Exception:
                    pass
        if hasattr(trainer, "loss_items"):
            try:
                li = trainer.loss_items
                if torch.is_tensor(li):
                    li = li.detach().cpu().tolist()
                return float(sum(map(float, li)))
            except Exception:
                pass
        m = getattr(trainer, "metrics", None)
        if isinstance(m, dict):
            for k in ("train/loss", "loss", "metrics/loss"):
                if k in m:
                    try:
                        return float(m[k])
                    except Exception:
                        pass
        return None

    def _get_total_epochs(self, trainer) -> Optional[int]:
        if hasattr(trainer, "epochs"):
            try:
                return int(getattr(trainer, "epochs"))
            except Exception:
                pass
        args = getattr(trainer, "args", None)
        if args is not None and hasattr(args, "epochs"):
            try:
                return int(getattr(args, "epochs"))
            except Exception:
                pass
        return None

    def _print_history(self):
        if not self.history:
            print("[History] No eval records.")
            return
        print("\n==== Eval History (epoch | loss | AP50(main) | LRs | LR reduced?) ====")
        for r in self.history:
            loss_s = (
                "NA"
                if r["train_loss"] is None or not math.isfinite(r["train_loss"])
                else f"{r['train_loss']:.6f}"
            )
            ap_s = f"{r['ap50']:.6f}"
            lr_s = (
                "NA"
                if r["lrs"] is None
                else "[" + ", ".join(f"{x:.6g}" for x in r["lrs"]) + "]"
            )
            red_s = "Y" if r["reduced"] else "N"
            print(
                f"epoch {r['epoch']:03d} | loss={loss_s} | AP50={ap_s} | "
                f"LRs={lr_s} | reduced={red_s}"
            )
        print("================================================================")

        # Print test summary
        if self.test_ap_history:
            best = max(self.test_ap_history, key=lambda d: d["ap50"])
            print("\n==== Periodic TEST AP50 (every N epochs) ====")
            for t in self.test_ap_history:
                print(f"epoch {t['epoch']:03d} | AP50(test)={t['ap50']:.6f}")
            print(
                f"Best TEST AP50={best['ap50']:.6f} at epoch {best['epoch']}"
            )
            print("=============================================\n")
        else:
            print("\n[TEST] No periodic test evaluations recorded.\n")

    def on_fit_end(self, trainer):
        """Callback hook for Ultralytics ``on_fit_end`` event."""
        if not self._printed:
            self._print_history()
            self._printed = True

    def on_train_end(self, trainer):
        """Callback hook for Ultralytics ``on_train_end`` event."""
        if not self._printed:
            self._print_history()
            self._printed = True
