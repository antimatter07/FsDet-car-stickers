import json, time, os, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Optional


def evaluate_fn_tags_visualize(
    coco_split_dir,
    predictions,
    class_map=None,
    yolo_names=None,
    save_dir="./coco_eval_out",
    target_size=(1280, 800),
    tags_of_interest=("left", "right", "far", "near"),
    tags_coco_json=None,
    # visualization params (matplotlib-based)
    visualize_img_indices=None,
    save_visuals=False,
    visual_save_dir=None,
    visual_conf_thresh=0.0,
    visual_box_alpha=0.45,
    visual_text_alpha=0.35,
    vis_limit=10,
    # toggle for side recall computation
    side_recall_mode="manual",  # "manual" or "coco"
    recall_conf_thresh=None,     # used in manual mode only. None = no filter
):
    """
    Evaluate detector outputs on a COCO style split with tag based breakdowns,
    side aware metrics, and optional visualizations.

    This function extends a standard COCO evaluation with:
      * Overall AP and AR on the the full split.
      * Per tag AP and AR for tags such as "left", "right", "far", "near".
      * Side aware AP for left and right halves of the windshield, using the
        largest windshield box in ground truth to define the midline.
      * Per tag side recall at IoU 0.5 for left and right halves, plus an
        overall recall that combines both sides.
      * Optional matplotlib based visualizations that highlight:
          - True positives in green.
          - False positives in orange.
          - Missed ground truth boxes in red.
          - Windshield boxes in purple.
        Visualizations can be shown inline or saved to disk.

    The function expects predictions in one of these formats:

      1) Dict indexed by image path:

         {
             "<image_path>": [
                 {"xyxy": [x1, y1, x2, y2], "conf": float, "cls": int},
                 ...
             ],
             ...
         }

      2) List of entries with explicit detection lists:

         [
             {
                 "image_path": "<path>",
                 "detections": [
                     {"xyxy": [...], "conf": float, "cls": int},
                     ...
                 ]
             },
             ...
         ]

         or with the key "boxes" instead of "detections".

    Ground truth boxes and prediction boxes are rescaled to a fixed reference
    size given by ``target_size`` before running COCOeval. Tag information is
    loaded from a separate COCO style JSON file that stores per image
    ``extra.user_tags`` or ``extra.tags`` lists.

    Args:
        coco_split_dir (str or Path):
            Directory containing the COCO split with ``_annotations.coco.json``
            and the corresponding images.
        predictions (dict or list):
            Detector predictions in one of the formats described above.
        class_map (dict, optional):
            Mapping from YOLO class indices to COCO category ids. If None, the
            function tries to infer it from ``yolo_names`` or by searching for
            a category whose name contains "sticker". If the split only has a
            single category, that category is used by default.
        yolo_names (list of str, optional):
            YOLO class names in index order. Used to construct ``class_map``
            automatically when possible.
        save_dir (str or Path, optional):
            Directory where the COCO detection JSON and optional visualizations
            will be written. Created if it does not exist.
        target_size (tuple[int, int], optional):
            Target image size as ``(width, height)`` used to rescale both
            ground truth and predictions before COCOeval.
        tags_of_interest (iterable of str, optional):
            Tag names that define the tag specific subsets, for example
            ``("left", "right", "far", "near")``. Matching is done on the
            lowercased tag strings.
        tags_coco_json (str or Path, optional):
            Path to a COCO style JSON file that contains per image
            ``extra.user_tags`` or ``extra.tags`` fields. These tags are used
            to assign each image to the tag subsets. If None, only the main
            COCO split is used and tag based subsets will be empty.
        visualize_img_indices (list[int], optional):
            If given, attempt to visualize only images whose file names start
            with any of the integer prefixes in this list (for example
            ``[149, 164, 194]``). Indices are matched to file names by prefix.
            If None, the function selects up to ``vis_limit`` images that have
            at least one true positive, false positive, or false negative.
        save_visuals (bool, optional):
            If True, save visualization figures as PNG files. If False, show
            them interactively with ``plt.show()``.
        visual_save_dir (str or Path, optional):
            Target directory for saved visualizations. If None and
            ``save_visuals`` is True, the directory
            ``save_dir / "visualizations"`` is used.
        visual_conf_thresh (float, optional):
            Minimum detection confidence required for a prediction to appear in
            the visualization panes. Has no effect on COCO metrics, only on
            what is drawn.
        visual_box_alpha (float, optional):
            Alpha value for the bounding box outlines in the visualizations.
            Used for all true positive, false positive, and false negative
            rectangles.
        visual_text_alpha (float, optional):
            Alpha value for the text background in the label boxes drawn on
            top of each rectangle.
        vis_limit (int, optional):
            Maximum number of images to visualize when
            ``visualize_img_indices`` is None.
        side_recall_mode (str, optional):
            Strategy used for per tag side recall at IoU 0.5:

              * "manual" - Uses a simple greedy matching between ground truth
                and detections based on IoU, and counts true positives and
                false negatives directly.
              * "coco"   - Builds a side specific COCO subset and uses
                COCOeval internals at IoU 0.5 to derive recall.

        recall_conf_thresh (float or None, optional):
            Confidence threshold used in the "manual" side recall mode to
            filter detections before matching. If None, all detections are
            considered.

    Returns:
        dict:
            A nested result dictionary with keys:

            ``"overall"``:
                Summary for the full split with keys:

                * ``"stats"``: dict of COCO metrics, including
                  ``"AP@[.5:.95]"``, ``"AP@0.50"``, ``"AP@0.75"``, and AR
                  values.
                * ``"n_images"``: number of images in this subset.
                * ``"n_gt"``: number of ground truth instances for the
                  evaluated classes.
                * ``"n_dt"``: number of detections for the evaluated classes.

            One entry per tag in ``tags_of_interest`` (lowercased):
                Same structure as ``"overall"``, but computed only on images
                that contain that tag.

            ``"left_side"`` and ``"right_side"``:
                Side aware COCO metrics restricted to the appropriate half of
                the windshield, if a windshield category can be identified.
                Each entry has the same structure as ``"overall"``. If no
                windshield category is found, these entries contain zero
                counts and ``stats=None``.

            ``"per_tag_side_recall"``:
                Nested dict indexed first by tag (lowercased) and then by side:

                * ``results["per_tag_side_recall"][tag]["left"]``:
                    Manual or COCO based recall at IoU 0.5 for the left half.
                * ``results["per_tag_side_recall"][tag]["right"]``:
                    Same for the right half.
                * ``results["per_tag_side_recall"][tag]["overall"]``:
                    Combined recall across both halves.

                Each side dictionary contains:

                  * ``"recall@0.5"``: recall value at IoU 0.5.
                  * ``"TP"``: number of true positives.
                  * ``"FN"``: number of false negatives.
                  * ``"n_images"``: number of images that contributed ground
                    truth or detections for that tag and side.
                  * ``"n_gt"``: total number of ground truth boxes considered.

            ``"AP50_overall"``:
                Convenience shortcut for
                ``results["overall"]["stats"]["AP@0.50"]``.

    Notes:
        * The function modifies the COCO ground truth annotations in place by
          rescaling bounding boxes and updating image sizes to match
          ``target_size``.
        * Side aware computations rely on a "windshield" category being
          present in the COCO categories. If it is missing, side based AP and
          side based recall are skipped.
        * Visualization requires matplotlib and PIL. If these imports fail,
          evaluation still runs but visualizations are skipped.
    """

    # -------------- utils --------------
    def _ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _largest_bbox(anns_list):
        if not anns_list:
            return None
        best, best_area = None, -1.0
        for a in anns_list:
            x, y, w, h = a["bbox"]
            area = w * h
            if area > best_area:
                best_area, best = area, a
        return best

    def _resolve_image_path(split_dir: Path, file_name: str) -> Optional[Path]:
        p = Path(file_name)
        cand = []
        if p.is_absolute():
            cand.append(p)
        cand.append(split_dir / p)
        cand.append(split_dir / "images" / p)
        cand.append(split_dir / "images" / p.name)
        if "images" in p.parts:
            cand.append(split_dir / "images" / p.name)
        for c in cand:
            if c.exists():
                return c
        return None

    # IoU helpers for manual recall
    def _xywh_to_xyxy(box):
        x, y, w, h = box
        return (x, y, x + w, y + h)

    def _iou(a_xywh, b_xywh):
        ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a_xywh)
        bx1, by1, bx2, by2 = _xywh_to_xyxy(b_xywh)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        denom = a_area + b_area - inter
        return inter / denom if denom > 0 else 0.0

    def _greedy_match_tp(gt_boxes, dt_boxes, iou_thr=0.5):
        # dt_boxes is list of (bbox, score)
        if not gt_boxes:
            return 0, 0
        dt_sorted = sorted(dt_boxes, key=lambda z: float(z[1]), reverse=True)
        gt_used = [False] * len(gt_boxes)
        tp = 0
        for db, _sc in dt_sorted:
            best_iou, best_idx = 0.0, -1
            for i, gb in enumerate(gt_boxes):
                if gt_used[i]:
                    continue
                iou = _iou(gb, db)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_thr and best_idx >= 0:
                gt_used[best_idx] = True
                tp += 1
        fn = len(gt_boxes) - tp
        return tp, fn

    try:
        import cv2  # optional
    except Exception:
        cv2 = None

    # -------------- load GT --------------
    coco_split_dir = Path(coco_split_dir)
    ann_path = coco_split_dir / "_annotations.coco.json"
    cocoGt = COCO(str(ann_path))
    img_ids = cocoGt.getImgIds()
    imgs = cocoGt.loadImgs(img_ids)

    basename_to_imgid = {Path(im["file_name"]).name: im["id"] for im in imgs}
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    name_to_catid = {c["name"]: c["id"] for c in cats}

    # meta for sub-COCOs
    info_meta = cocoGt.dataset.get("info", {})
    licenses_meta = cocoGt.dataset.get("licenses", [])

    def _build_coco_subset(images_subset, anns_subset, use_eval_cats=True):
        sub = COCO()
        sub.dataset = {
            "info": info_meta,
            "licenses": licenses_meta,
            "images": images_subset,
            "annotations": anns_subset,
            "categories": ([c for c in cats if c["id"] in eval_cat_ids] if use_eval_cats else cats),
        }
        sub.createIndex()
        return sub

    # -------------- class_map --------------
    if class_map is None:
        if yolo_names:
            tmp = {i: name_to_catid[nm] for i, nm in enumerate(yolo_names) if nm in name_to_catid}
            if tmp:
                class_map = tmp
        if class_map is None:
            sticker_id = next((c["id"] for c in cats if "sticker" in c["name"].lower()), None)
            if sticker_id:
                class_map = {0: sticker_id}
    if class_map is None and len(cats) == 1:
        class_map = {0: cats[0]["id"]}
    if class_map is None:
        raise ValueError("[evaluate] Could not infer class_map on multi-class COCO.")
    eval_cat_ids = sorted(set(class_map.values()))

    def to_cat_id(yolo_cls):
        return class_map.get(int(yolo_cls), eval_cat_ids[0])

    # windshield cat (optional)
    windshield_cat_id = None
    if yolo_names and "windshield" in yolo_names and "windshield" in name_to_catid:
        windshield_cat_id = name_to_catid["windshield"]
    if windshield_cat_id is None:
        for c in cats:
            if c["name"].lower() == "windshield":
                windshield_cat_id = c["id"]
                break

    # for viz color override
    windshield_cat_ids = {c["id"] for c in cats if "windshield" in c["name"].lower()}

    target_w, target_h = target_size

    # -------------- scale GT to target (in-place) --------------
    imgid_to_origsz = {im["id"]: (im["width"], im["height"]) for im in imgs}
    for im in imgs:
        ow, oh = im["width"], im["height"]
        sx, sy = target_w / ow, target_h / oh
        ann_ids_img = cocoGt.getAnnIds(imgIds=[im["id"]])
        for ann in cocoGt.loadAnns(ann_ids_img):
            x, y, w, h = ann["bbox"]
            ann["bbox"] = [x * sx, y * sy, w * sx, h * sy]
        im["width"], im["height"] = target_w, target_h

    # -------------- normalize predictions --------------
    norm_preds = []
    if isinstance(predictions, dict):
        for k, v in predictions.items():
            norm_preds.append({"image_path": k, "detections": v})
    elif isinstance(predictions, list):
        for it in predictions:
            dets = it.get("detections", it.get("boxes", []))
            ip = it.get("orig_path") or it.get("image_path") or it.get("crop_path")
            norm_preds.append({"image_path": ip, "detections": dets})
    else:
        raise TypeError("predictions must be dict or list")

    # -------------- build detection list (scaled) --------------
    dt_list, skipped = [], 0
    for it in norm_preds:
        ipath = it["image_path"]
        if not ipath:
            continue
        base = Path(ipath).name
        img_id = basename_to_imgid.get(base)
        if img_id is None:
            skipped += 1
            continue
        ow, oh = imgid_to_origsz[img_id]
        sx, sy = target_w / ow, target_h / oh
        for d in it.get("detections", []):
            x1, y1, x2, y2 = d["xyxy"]
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            dt_list.append({
                "image_id": int(img_id),
                "category_id": int(to_cat_id(d.get("cls", 0))),
                "bbox": [float(x1), float(y1), w, h],
                "score": float(d.get("conf", 0.0))
            })
    if skipped:
        print(f"[evaluate] Skipped {skipped} prediction entries not found in COCO GT by basename.")

    for cid in eval_cat_ids:
        n_gt = len(cocoGt.getAnnIds(catIds=[cid]))
        n_dt = sum(1 for d in dt_list if d["category_id"] == cid)
        cname = next((c["name"] for c in cats if c["id"] == cid), str(cid))
        print(f"[evaluate] Category '{cname}' (id={cid}): GT={n_gt}, DT={n_dt}")

    # -------------- COCO results --------------
    save_dir = Path(save_dir)
    _ensure_dir(save_dir)
    dt_json = save_dir / f"coco_dt_{int(time.time())}.json"
    dt_json.write_text(json.dumps(dt_list))
    cocoDt = cocoGt.loadRes(str(dt_json)) if len(dt_list) > 0 else COCO()

    metric_names = [
        "AP@[.5:.95]", "AP@0.50", "AP@0.75", "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large"
    ]

    def run_eval(subset_img_ids, label="overall", coco_gt=None, coco_dt=None):
        if not subset_img_ids:
            print(f"[evaluate:{label}] No images in subset.")
            return {"stats": None, "n_images": 0, "n_gt": 0, "n_dt": 0}
        _coco_gt = coco_gt if coco_gt is not None else cocoGt
        _coco_dt = coco_dt if coco_dt is not None else cocoDt
        E = COCOeval(_coco_gt, _coco_dt, iouType="bbox")
        E.params.imgIds = list(subset_img_ids)
        E.params.catIds = eval_cat_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
        n_gt = sum(len(_coco_gt.getAnnIds(imgIds=[iid], catIds=eval_cat_ids)) for iid in subset_img_ids)
        n_dt = 0
        if hasattr(_coco_dt, "anns") and _coco_dt.anns:
            for _, ann in _coco_dt.anns.items():
                if ann.get("image_id") in subset_img_ids and ann.get("category_id") in eval_cat_ids:
                    n_dt += 1
        return {
            "stats": {k: float(E.stats[i]) for i, k in enumerate(metric_names)},
            "n_images": len(subset_img_ids),
            "n_gt": int(n_gt),
            "n_dt": int(n_dt),
        }

    # -------------- tags ingestion --------------
    tags_map = {}
    if tags_coco_json:
        tdata = json.loads(Path(tags_coco_json).read_text())
        for im in tdata.get("images", []):
            keys = set()
            fn = im.get("file_name") or ""
            en = (im.get("extra") or {}).get("name") or ""
            for s in (fn, Path(fn).name, Path(fn).stem, en, Path(en).name, Path(en).stem):
                if s:
                    keys.add(str(s))
            raw = (im.get("extra") or {}).get("user_tags") or (im.get("extra") or {}).get("tags") or []
            if not isinstance(raw, list):
                raw = [raw]
            tags = [t.strip().lower() for t in raw if isinstance(t, str) and t.strip()]
            for k in keys:
                tags_map[k] = tags

    imgid_to_tags = {}
    for im in imgs:
        keys = [
            im.get("file_name") or "",
            Path(im.get("file_name") or "").name,
            Path(im.get("file_name") or "").stem,
        ]
        tags = []
        for k in keys:
            if k in tags_map:
                tags = tags_map[k]
                break
        imgid_to_tags[im["id"]] = tags

    # -------------- overall + per-tag AP and AR --------------
    results = {}
    results["overall"] = run_eval(img_ids, "overall")
    tags_lower = [t.lower() for t in tags_of_interest]
    for t in tags_lower:
        subset = [iid for iid, tl in imgid_to_tags.items() if t in tl]
        print(f"[evaluate] Tag '{t}': {len(subset)} images.")
        results[t] = run_eval(subset, t)
    results["AP50_overall"] = results["overall"]["stats"]["AP@0.50"] if results["overall"]["stats"] else None

    # -------------- side-aware AP for left and right images --------------
    if windshield_cat_id is not None:
        imgid_to_ws = {}
        for iid in img_ids:
            ws_ids = cocoGt.getAnnIds(imgIds=[iid], catIds=[windshield_cat_id])
            if not ws_ids:
                continue
            best = _largest_bbox(cocoGt.loadAnns(ws_ids))
            if not best:
                continue
            bx, by, bw, bh = best["bbox"]
            xmid = bx + bw * 0.5
            imgid_to_ws[iid] = {"bbox": [bx, by, bw, bh], "xmid": xmid}

        left_tag_imgs = {iid for iid, tags in imgid_to_tags.items() if "left" in tags}
        right_tag_imgs = {iid for iid, tags in imgid_to_tags.items() if "right" in tags}
        left_img_ids_ws = sorted(i for i in left_tag_imgs if i in imgid_to_ws)
        right_img_ids_ws = sorted(i for i in right_tag_imgs if i in imgid_to_ws)

        def _collect_side_gt(img_id_list, side: str):
            out = []
            for iid in img_id_list:
                meta = imgid_to_ws[iid]
                xmid = meta["xmid"]
                wx, wy, ww, wh = meta["bbox"]
                for a in cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[iid], catIds=eval_cat_ids)):
                    x, y, w, h = a["bbox"]
                    cx = x + w * 0.5
                    if not (wx <= cx <= wx + ww):
                        continue
                    if (side == "left" and cx <= xmid) or (side == "right" and cx > xmid):
                        out.append(a)
            return out

        def _collect_side_dt(img_id_list, side: str):
            out = []
            for iid in img_id_list:
                meta = imgid_to_ws.get(iid)
                if not meta:
                    continue
                xmid = meta["xmid"]
                wx, wy, ww, wh = meta["bbox"]
                for d in dt_list:
                    if d["image_id"] != iid:
                        continue
                    if d["category_id"] not in eval_cat_ids:
                        continue
                    x, y, w, h = d["bbox"]
                    cx = x + w * 0.5
                    if not (wx <= cx <= wx + ww):
                        continue
                    if (side == "left" and cx <= xmid) or (side == "right" and cx > xmid):
                        out.append(d)
            return out

        left_gt_anns = _collect_side_gt(left_img_ids_ws, "left")
        left_imgs = [im for im in imgs if im["id"] in left_img_ids_ws]
        coco_left = _build_coco_subset(left_imgs, left_gt_anns)
        left_dt = _collect_side_dt(left_img_ids_ws, "left")
        cocoDt_left = coco_left.loadRes(left_dt) if left_dt else COCO()
        results["left_side"] = run_eval(set(left_img_ids_ws), "left_side", coco_gt=coco_left, coco_dt=cocoDt_left)

        right_gt_anns = _collect_side_gt(right_img_ids_ws, "right")
        right_imgs = [im for im in imgs if im["id"] in right_img_ids_ws]
        coco_right = _build_coco_subset(right_imgs, right_gt_anns)
        right_dt = _collect_side_dt(right_img_ids_ws, "right")
        cocoDt_right = coco_right.loadRes(right_dt) if right_dt else COCO()
        results["right_side"] = run_eval(set(right_img_ids_ws), "right_side", coco_gt=coco_right, coco_dt=cocoDt_right)
    else:
        print("[evaluate][side-aware] No 'windshield' category; skipping left and right side AP.")
        results["left_side"] = {"stats": None, "n_images": 0, "n_gt": 0, "n_dt": 0}
        results["right_side"] = {"stats": None, "n_images": 0, "n_gt": 0, "n_dt": 0}

    # -------------- per-tag side-wise recall@0.5 --------------
    results["per_tag_side_recall"] = {}
    if windshield_cat_id is not None:
        # build once
        imgid_to_ws = {}
        for iid in img_ids:
            ws_ids = cocoGt.getAnnIds(imgIds=[iid], catIds=[windshield_cat_id])
            if not ws_ids:
                continue
            best = _largest_bbox(cocoGt.loadAnns(ws_ids))
            if not best:
                continue
            bx, by, bw, bh = best["bbox"]
            xmid = bx + bw * 0.5
            imgid_to_ws[iid] = {"bbox": [bx, by, bw, bh], "xmid": xmid}

        # index GT and DT per image
        gt_by_img = {
            iid: [a for a in cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[iid], catIds=eval_cat_ids))]
            for iid in img_ids
        }
        dt_by_img = defaultdict(list)
        for d in dt_list:
            if d["category_id"] in eval_cat_ids:
                dt_by_img[d["image_id"]].append(d)

        def _side_boxes_from_anns(iid, side, anns, ws_meta):
            out = []
            xmid = ws_meta["xmid"]
            wx, wy, ww, wh = ws_meta["bbox"]
            for a in anns:
                x, y, w, h = a["bbox"]
                cx = x + w * 0.5
                if not (wx <= cx <= wx + ww):
                    continue
                if (side == "left" and cx <= xmid) or (side == "right" and cx > xmid):
                    out.append(a)
            return out

        def _collect_gt_dt_for_tag_side_manual(tag_str, side):
            all_gt_boxes = []
            all_dt_boxes = []
            contributing_img_ids = set()
            for iid, tags in imgid_to_tags.items():
                if tag_str not in tags:
                    continue
                if iid not in imgid_to_ws:
                    continue
                ws_meta = imgid_to_ws[iid]
                # GT side
                gt_side_anns = _side_boxes_from_anns(iid, side, gt_by_img[iid], ws_meta)
                if gt_side_anns:
                    contributing_img_ids.add(iid)
                all_gt_boxes.extend([g["bbox"] for g in gt_side_anns])
                # DT side
                dts = dt_by_img.get(iid, [])
                if recall_conf_thresh is None:
                    keep = dts
                else:
                    keep = [d for d in dts if float(d.get("score", 0.0)) >= float(recall_conf_thresh)]
                dt_side = _side_boxes_from_anns(iid, side, keep, ws_meta)
                if dt_side:
                    contributing_img_ids.add(iid)
                all_dt_boxes.extend([(d["bbox"], float(d.get("score", 0.0))) for d in dt_side])
            return all_gt_boxes, all_dt_boxes, contributing_img_ids

        def _collect_gt_dt_for_tag_side_coco(tag_str, side):
            imgs_sub = []
            gt_anns = []
            dt_anns = []
            contributing_img_ids = set()
            for iid, tags in imgid_to_tags.items():
                if tag_str not in tags:
                    continue
                if iid not in imgid_to_ws:
                    continue
                ws_meta = imgid_to_ws[iid]
                gt_side_anns = _side_boxes_from_anns(iid, side, gt_by_img[iid], ws_meta)
                dts = dt_by_img.get(iid, [])
                dt_side = _side_boxes_from_anns(iid, side, dts, ws_meta)
                if gt_side_anns or dt_side:
                    imgs_sub.append(next(i for i in imgs if i["id"] == iid))
                    contributing_img_ids.add(iid)
                gt_anns.extend(gt_side_anns)
                dt_anns.extend(dt_side)
            coco_sub = _build_coco_subset(imgs_sub, gt_anns)
            cocoDt_sub = coco_sub.loadRes(dt_anns) if dt_anns else COCO()
            return coco_sub, cocoDt_sub, [im["id"] for im in imgs_sub], contributing_img_ids

        def _recall_at_05_coco(coco_sub, cocoDt_sub, img_ids_sub):
            if not img_ids_sub:
                return 0.0, 0, 0
            E = COCOeval(coco_sub, cocoDt_sub, iouType="bbox")
            E.params.imgIds = list(img_ids_sub)
            E.params.catIds = eval_cat_ids
            E.evaluate()
            E.accumulate()
            iou_thrs = E.params.iouThrs
            t_idx = int(np.argmin(np.abs(iou_thrs - 0.5)))
            total_gt, matched_gt = 0, 0
            for ei in (E.evalImgs or []):
                if ei is None or ei.get("category_id") not in eval_cat_ids:
                    continue
                gtIds = ei.get("gtIds", [])
                gtMatches = ei.get("gtMatches")
                gtIgnore = ei.get("gtIgnore", np.zeros(len(gtIds), dtype=bool))
                if gtMatches is None:
                    continue
                gtMatches = np.array(gtMatches)
                gtIgnore = np.array(gtIgnore)
                if gtMatches.shape[1] != len(gtIds):
                    continue
                rowg = gtMatches[t_idx]
                for k, gid in enumerate(gtIds):
                    if gtIgnore[k]:
                        continue
                    total_gt += 1
                    if rowg[k] > 0:
                        matched_gt += 1
            recall = float(matched_gt) / float(total_gt) if total_gt > 0 else 0.0
            return recall, matched_gt, total_gt

        for tag in tags_lower:
            results["per_tag_side_recall"].setdefault(tag, {})

            if side_recall_mode.lower() == "manual":
                # left
                gt_l, dt_l, ids_l = _collect_gt_dt_for_tag_side_manual(tag, "left")
                tp_l, fn_l = _greedy_match_tp(gt_l, dt_l, iou_thr=0.5)
                tot_l = tp_l + fn_l
                r_l = (tp_l / tot_l) if tot_l > 0 else 0.0
                # right
                gt_r, dt_r, ids_r = _collect_gt_dt_for_tag_side_manual(tag, "right")
                tp_r, fn_r = _greedy_match_tp(gt_r, dt_r, iou_thr=0.5)
                tot_r = tp_r + fn_r
                r_r = (tp_r / tot_r) if tot_r > 0 else 0.0
            else:  # "coco"
                coco_l, cocoDt_l, img_ids_l, ids_l = _collect_gt_dt_for_tag_side_coco(tag, "left")
                r_l, tp_l, tot_l = _recall_at_05_coco(coco_l, cocoDt_l, img_ids_l)
                coco_r, cocoDt_r, img_ids_r, ids_r = _collect_gt_dt_for_tag_side_coco(tag, "right")
                r_r, tp_r, tot_r = _recall_at_05_coco(coco_r, cocoDt_r, img_ids_r)

            n_img_tag_ws = len(ids_l.union(ids_r))

            # store left and right
            results["per_tag_side_recall"][tag]["left"] = {
                "recall@0.5": r_l,
                "TP": tp_l,
                "FN": tot_l - tp_l if side_recall_mode == "coco" else fn_l,
                "n_images": len(ids_l),
                "n_gt": tot_l,
            }
            results["per_tag_side_recall"][tag]["right"] = {
                "recall@0.5": r_r,
                "TP": tp_r,
                "FN": tot_r - tp_r if side_recall_mode == "coco" else fn_r,
                "n_images": len(ids_r),
                "n_gt": tot_r,
            }

            # overall across left and right
            tp_all = tp_l + tp_r
            gt_all = tot_l + tot_r
            fn_all = gt_all - tp_all
            r_all = (tp_all / gt_all) if gt_all > 0 else 0.0
            results["per_tag_side_recall"][tag]["overall"] = {
                "recall@0.5": r_all,
                "TP": tp_all,
                "FN": fn_all,
                "n_images": n_img_tag_ws,
                "n_gt": gt_all,
            }
    else:
        print("[evaluate][per-tag side recall] No 'windshield' category; skipping.")

    # -------------- summary print --------------
    print("\n=== EVAL SUMMARY ===")

    def _line(lbl, d):
        if not d["stats"]:
            return f"{lbl:>12}: n_images={d['n_images']} | n_gt={d['n_gt']} | n_dt={d['n_dt']} (no stats)"
        return (
            f"{lbl:>12}: n_images={d['n_images']} | n_gt={d['n_gt']} | n_dt={d['n_dt']} | "
            f"AP50={d['stats']['AP@0.50']:.4f} | AP={d['stats']['AP@[.5:.95]']:.4f}"
        )

    print(_line("overall", results["overall"]))
    for t in tags_lower:
        print(_line(t, results[t]))
    if "left_side" in results:
        print(_line("left_side", results["left_side"]))
    if "right_side" in results:
        print(_line("right_side", results["right_side"]))
    if "per_tag_side_recall" in results:
        print("\n=== PER-TAG SIDE RECALL @0.5 ===")
        for tag, sides in results["per_tag_side_recall"].items():
            l = sides.get("left", {"recall@0.5": 0.0, "n_gt": 0, "n_images": 0, "TP": 0})
            r = sides.get("right", {"recall@0.5": 0.0, "n_gt": 0, "n_images": 0, "TP": 0})
            o = sides.get("overall", {"recall@0.5": 0.0, "n_gt": 0, "n_images": 0, "TP": 0})
            print(
                f"{tag:>8} | left:  R={l['recall@0.5']:.4f} (TP={l['TP']}/GT={l['n_gt']}, imgs={l['n_images']})"
                f"   right: R={r['recall@0.5']:.4f} (TP={r['TP']}/GT={r['n_gt']}, imgs={r['n_images']})"
                f"   overall: R={o['recall@0.5']:.4f} (TP={o['TP']}/GT={o['n_gt']}, imgs={o['n_images']})"
            )

    # -------------- visualization (matplotlib) --------------
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image

        Eall = COCOeval(cocoGt, cocoDt, iouType="bbox")
        Eall.params.imgIds = list(img_ids)
        Eall.params.catIds = eval_cat_ids
        Eall.evaluate()
        Eall.accumulate()

        iou_thrs = Eall.params.iouThrs
        t_idx = int(np.argmin(np.abs(iou_thrs - 0.5)))

        tp_per_img = defaultdict(list)  # (bbox, cat_id, score)
        fp_per_img = defaultdict(list)  # (bbox, cat_id, score)
        fn_per_img = defaultdict(list)  # (bbox, cat_id)

        dt_anns = {
            a["id"]: a
            for a in (
                cocoDt.loadAnns(cocoDt.getAnnIds())
                if hasattr(cocoDt, "dataset") and cocoDt.dataset
                else []
            )
        }
        gt_anns = {a["id"]: a for a in cocoGt.loadAnns(cocoGt.getAnnIds())}

        for ev in (Eall.evalImgs or []):
            if ev is None:
                continue
            img_id = ev["image_id"]
            if ev.get("category_id") not in eval_cat_ids:
                continue

            dtIds = ev.get("dtIds", [])
            gtIds = ev.get("gtIds", [])
            if len(dtIds) == 0 and len(gtIds) == 0:
                continue

            dtMatches = np.array(ev.get("dtMatches", []))
            gtMatches = np.array(ev.get("gtMatches", []))
            dtIgnore = np.array(ev.get("dtIgnore", np.zeros(len(dtIds), dtype=bool)))
            gtIgnore = np.array(ev.get("gtIgnore", np.zeros(len(gtIds), dtype=bool)))

            # detections
            if dtIds is not None:
                is_2d = dtMatches.ndim == 2 and dtMatches.shape[0] == len(iou_thrs)
                for j, dt_id in enumerate(dtIds):
                    ann = dt_anns.get(dt_id)
                    if not ann:
                        continue
                    score = float(ann.get("score", 0.0))
                    if score < visual_conf_thresh:
                        continue
                    if is_2d:
                        matched_gt_id = int(dtMatches[t_idx, j]) if j < dtMatches.shape[1] else 0
                        is_ignored = (
                            bool(dtIgnore[t_idx, j])
                            if dtIgnore.ndim == 2
                            else bool(dtIgnore[j])
                            if dtIgnore.size > j
                            else False
                        )
                    else:
                        matched_gt_id = int(dtMatches[j]) if dtMatches.size > j else 0
                        is_ignored = bool(dtIgnore[j]) if dtIgnore.size > j else False
                    if is_ignored:
                        continue
                    bbox = ann["bbox"]
                    cat_id = ann["category_id"]
                    if matched_gt_id > 0:
                        tp_per_img[img_id].append((bbox, cat_id, score))
                    else:
                        fp_per_img[img_id].append((bbox, cat_id, score))

            # ground truths
            if gtIds is not None:
                is_2d = gtMatches.ndim == 2 and gtMatches.shape[0] == len(iou_thrs)
                for j, gt_id in enumerate(gtIds):
                    ann = gt_anns.get(gt_id)
                    if not ann:
                        continue
                    if is_2d:
                        matched_dt_id = int(gtMatches[t_idx, j]) if j < gtMatches.shape[1] else 0
                        is_ignored = (
                            bool(gtIgnore[t_idx, j])
                            if gtIgnore.ndim == 2
                            else bool(gtIgnore[j])
                            if gtIgnore.size > j
                            else False
                        )
                    else:
                        matched_dt_id = int(gtMatches[j]) if gtMatches.size > j else 0
                        is_ignored = bool(gtIgnore[j]) if gtIgnore.size > j else False
                    if is_ignored:
                        continue
                    if matched_dt_id == 0:
                        bbox = ann["bbox"]
                        cat_id = ann["category_id"]
                        fn_per_img[img_id].append((bbox, cat_id))

        # choose images
        if visualize_img_indices is not None:
            candidate_imgs = []
            for idx in visualize_img_indices:
                idx_str = str(idx)
                prefix1 = f"{idx_str}_jpg"
                prefix2 = f"{idx_str}_"
                matched_img_id = None
                for base, _img_id in basename_to_imgid.items():
                    if base.startswith(prefix1) or base.startswith(prefix2):
                        matched_img_id = _img_id
                        break
                if matched_img_id is not None and matched_img_id not in candidate_imgs:
                    candidate_imgs.append(matched_img_id)

            # keep only ones that have any TP, FP, or FN at the applied thresholds
            candidate_imgs = [
                img_id
                for img_id in candidate_imgs
                if tp_per_img[img_id] or fp_per_img[img_id] or fn_per_img[img_id]
            ]
            if not candidate_imgs:
                print(f"[viz] No images matched visualize_img_indices={visualize_img_indices} after thresholding.")
        else:
            # take first N in dataset order that have any TP, FP, or FN
            candidate_imgs = []
            for img_id in img_ids:
                if tp_per_img[img_id] or fp_per_img[img_id] or fn_per_img[img_id]:
                    candidate_imgs.append(img_id)
                if len(candidate_imgs) == vis_limit:
                    break

        if not candidate_imgs:
            print("[viz] No images to visualize.")
        else:
            if save_visuals:
                vis_dir = Path(visual_save_dir) if visual_save_dir else (save_dir / "visualizations")
                _ensure_dir(vis_dir)

            def get_color(base_color, cat_id):
                return "purple" if cat_id in windshield_cat_ids else base_color

            def draw_label(ax, x, y, text):
                ax.text(
                    x,
                    max(y - 2, 0),
                    text,
                    fontsize=6,
                    color="white",
                    alpha=visual_text_alpha,
                    ha="left",
                    va="bottom",
                    bbox=dict(
                        facecolor="black",
                        edgecolor="none",
                        pad=1.0,
                        alpha=visual_text_alpha,
                    ),
                )

            for img_id in candidate_imgs:
                img_info = cocoGt.loadImgs([img_id])[0]
                img_path = _resolve_image_path(coco_split_dir, img_info["file_name"])
                if img_path is None:
                    print(f"[viz] Missing image file for {img_info['file_name']} under {coco_split_dir}, skipping.")
                    continue
                try:
                    from PIL import Image
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"[viz] Could not load image for visualization: {img_path} ({e})")
                    continue
                img_resized = img.resize((target_w, target_h))

                fig = plt.figure(figsize=(8, 6))
                plt.imshow(img_resized)
                ax = plt.gca()
                ax.set_title(f"image_id={img_id}")
                ax.axis("off")

                # TP
                for (bbox, cat_id, score) in tp_per_img.get(img_id, []):
                    x, y, w, h = bbox
                    rect = patches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=1,
                        edgecolor=get_color("green", cat_id),
                        facecolor="none",
                        alpha=visual_box_alpha,
                    )
                    ax.add_patch(rect)
                    draw_label(ax, x, y, f"{score:.2f}")

                # FP
                for (bbox, cat_id, score) in fp_per_img.get(img_id, []):
                    x, y, w, h = bbox
                    rect = patches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=1,
                        edgecolor=get_color("orange", cat_id),
                        facecolor="none",
                        alpha=visual_box_alpha,
                    )
                    ax.add_patch(rect)
                    draw_label(ax, x, y, f"{score:.2f}")

                # FN
                for (bbox, cat_id) in fn_per_img.get(img_id, []):
                    x, y, w, h = bbox
                    rect = patches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=1,
                        edgecolor=get_color("red", cat_id),
                        facecolor="none",
                        alpha=visual_box_alpha,
                    )
                    ax.add_patch(rect)
                    draw_label(ax, x, y, "miss")

                if save_visuals:
                    out_name = f"vis_{img_id}.png"
                    out_path = (
                        Path(visual_save_dir)
                        if visual_save_dir
                        else (save_dir / "visualizations")
                    ) / out_name
                    fig.savefig(out_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    print(f"[viz] Wrote {out_path}")
                else:
                    plt.show()

    except ImportError as e:
        print(f"[viz] Visualization skipped (missing dependency): {e}")

    return results
