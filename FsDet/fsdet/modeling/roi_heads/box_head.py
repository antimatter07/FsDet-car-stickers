"""
box_head.py

Defines ROI box head modules for object detection.

This module provides a registry and implementations for ROI box heads,
which compute features from per-region inputs and optionally produce
box predictions. It includes:

- `FastRCNNConvFCHead`: A standard convolution + fully connected head.
- `ROI_BOX_HEAD_REGISTRY`: Registry for creating box heads dynamically.

The registered box head objects are constructed using:
    `obj(cfg, input_shape)` 
where `cfg` is a Detectron2 config object and `input_shape` describes
the input tensor dimensions.
"""
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        Initialize the FastRCNNConvFCHead.

        Parses configuration to determine the number of conv/fc layers,
        dimensions, and normalization.

        Args:
            cfg (CfgNode): Detectron2 configuration object.
            input_shape (ShapeSpec): Shape of the input tensor, including
                                     channels, height, and width.

        Attributes:
            conv_norm_relus (list[nn.Module]): List of convolution layers.
            fcs (list[nn.Module]): List of fully connected layers.
            _output_size (int or tuple): Output size of the head, either
                                         channels x height x width (before FCs)
                                         or feature dimension (after FCs).
        """
        super().__init__()

        # noqa: E221
        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (
                conv_dim,
                self._output_size[1],
                self._output_size[2],
            )

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        """
        Forward pass through conv + fc layers.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Output features after all conv and fc layers.
        """
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        """
        Output size of the head.

        Returns:
            int or tuple: Number of output channels x height x width (before FC)
                          or feature dimension (after FC layers).
        """
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head module according to the configuration.

    Args:
        cfg (CfgNode): Detectron2 configuration object.
        input_shape (ShapeSpec): Input shape of the ROI features.

    Returns:
        nn.Module: Instantiated ROI box head.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
