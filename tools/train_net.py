"""
Detection Training Script.

This script reads a given configuration file and runs training or evaluation
for standard models in FsDet.

It is designed to support multiple built-in models and datasets.
"""

import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    verify_results,
)

from fsdet.evaluation.stickers_evaluation import CarStickerEvaluator

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.

        This function uses the metadata field "evaluator_type" associated with
        each built-in dataset. For custom datasets, you can create an evaluator
        manually in your script.

        Args:
            cfg: FsDet/Detectron2 configuration object.
            dataset_name (str): Name of the dataset to evaluate.
            output_folder (str, optional): Directory to save evaluation results.
                Defaults to "<cfg.OUTPUT_DIR>/inference".

        Returns:
            evaluator: Evaluator object or DatasetEvaluators if multiple evaluators
                       are returned.

        Raises:
            NotImplementedError: If no evaluator is available for the dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "stickers":
            return CarStickerEvaluator(dataset_name, cfg, False, output_folder)
            
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configuration object and perform basic setup.

    Args:
        args: Parsed command line arguments.

    Returns:
        cfg: Frozen FsDet configuration object.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    """
    Main entry point for training or evaluation.

    Args:
        args: Parsed command line arguments.

    Returns:
        If eval_only is set, returns evaluation results dictionary.
        Otherwise, returns the result of trainer.train().
    """
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    print(trainer.model)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
