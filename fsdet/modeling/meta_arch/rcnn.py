import logging

import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from torch import nn
from detectron2.modeling.backbone import ResNet
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.backbone.resnet import build_resnet_backbone


from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone

import math


from fsdet.modeling.roi_heads import build_roi_heads

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Extended Generalized R-CNN architecture for FsDet.

    This implementation follows the standard Detectron2 two-stage pipeline,
    consisting of:

        1. Backbone – Extracts per-image convolutional feature maps.
        2. Proposal Generator (RPN) – Generates region proposals based on backbone features.
        3. ROI Heads – Performs feature pooling and predicts class logits & bounding boxes.

    This version includes additional capabilities:
        - Optional freezing of backbone, RPN, or ROI head feature layers.
        - Unified preprocessing and normalization.
        - Compatibility with FsDet's custom ROI heads and architectural overrides.

    Args:
        cfg (CfgNode):
            A Detectron2-style configuration node. Must provide model,
            input, and solver settings including PIXEL_MEAN, PIXEL_STD,
            backbone name, ROI head configuration, and device.

    Attributes:
        device (torch.device):
            Device on which the model is executed.
        backbone (nn.Module):
            CNN feature extractor module.
        proposal_generator (nn.Module):
            Region Proposal Network producing bounding box proposals.
        roi_heads (nn.Module):
            ROI heads responsible for classification and bounding box regression.
        normalizer (callable):
            Preprocessing function that normalizes images using PIXEL_MEAN/STD.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on batched inputs.

        Args:
            batched_inputs (list[dict]): list of inputs per image.
            detected_instances (None or list[Instances]): known detections per image. Skip proposal generation if provided.
            do_postprocess (bool): whether to postprocess outputs to original image resolution.

        Returns:
            list[dict]: results per image, each dict contains:
                - "instances": predicted Instances object with "pred_boxes", "pred_classes", "scores".
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad, and batch input images.

        Args:
            batched_inputs (list[dict]): list of dicts with key "image" per image.

        Returns:
            ImageList: batched images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    RPN-only network.

    Implements a backbone + region proposal generator architecture without ROI heads.

    Args:
        cfg (CfgNode): Detectron2 configuration.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

#@BACKBONE_REGISTRY.register()
#def build_dilated_resnet_fpn_backbone(cfg, input_shape):
    # Load default ResNet FPN backbone
 #   backbone = build_resnet_fpn_backbone(cfg, input_shape)
    
    # Modify stage 5 (`res5`) to be dilated
  #  for bottleneck in backbone.bottom_up.stages[3].children():
   #     bottleneck.conv2 = torch.nn.Conv2d(
    #        in_channels=bottleneck.conv2.in_channels,
     #       out_channels=bottleneck.conv2.out_channels,
      #      kernel_size=bottleneck.conv2.kernel_size,
     #       stride=1,  # Remove stride
     #       padding=2,  # Adjust padding
     #       dilation=2,  # Apply dilation
     #       bias=(bottleneck.conv2.bias is not None)  # Keep bias if it was present
     #   )

    #return backbone

