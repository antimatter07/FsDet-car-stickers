import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch
from detectron2.utils.comm import is_main_process
from detectron2.structures import Instances, Boxes
import copy


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(
                        k
                    )
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, cfg):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    if cfg.SPLICE:
        print('in splice evaluator!')
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()

            if cfg.SPLICE:
                # --- Begin splicing logic ---
               
                splice_outputs = []
                for input in inputs:
                    image = input["image"]
                    h, w = image.shape[1], image.shape[2]
            
                    tile_size = 512
                    stride = 384  # overlap to avoid edge-cut issues
            
                    tiles = []
                    tile_infos = []
            
                    for y0 in range(0, h, stride):
                        for x0 in range(0, w, stride):
                            y1 = min(y0 + tile_size, h)
                            x1 = min(x0 + tile_size, w)
            
                            tile = image[:, y0:y1, x0:x1]
            
                            tile_input = copy.deepcopy(input)
                            tile_input["image"] = tile
                            tile_input["height"] = y1 - y0
                            tile_input["width"] = x1 - x0
                            tile_infos.append((x0, y0, x1 - x0, y1 - y0))
                            tiles.append(tile_input)
            
                    tile_outputs = model(tiles)
            
                    merged_output = {"instances": []}
                    for out, (x0, y0, _, _) in zip(tile_outputs, tile_infos):
                        instances = out["instances"].to("cpu")
                        if instances.has("pred_boxes") and len(instances) > 0:
                            boxes = instances.pred_boxes.tensor.clone()
                            boxes[:, 0::2] += x0
                            boxes[:, 1::2] += y0
                            instances.pred_boxes = Boxes(boxes)
                            merged_output["instances"].append(instances)
            
                    if merged_output["instances"]:
                        merged_instances = Instances((h, w))
                        all_instances = merged_output["instances"]
            
                        merged_instances.pred_boxes = Boxes.cat([
                            i.pred_boxes for i in all_instances if i.has("pred_boxes")
                        ])
                        merged_instances.scores = torch.cat([
                            i.scores for i in all_instances if i.has("scores")
                        ])
                        merged_instances.pred_classes = torch.cat([
                            i.pred_classes for i in all_instances if i.has("pred_classes")
                        ])
                        output = {"instances": merged_instances}
                    else:
                        output = {"instances": Instances((h, w))}  # empty Instances
            
                    splice_outputs.append(output)
            
                outputs = splice_outputs
                # --- End splicing logic ---
            else:
                outputs = model(inputs)
            
            
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(
                        seconds_per_img * (total - num_warmup) - duration
                    )
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(
        datetime.timedelta(seconds=int(total_compute_time))
    )
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results



@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
