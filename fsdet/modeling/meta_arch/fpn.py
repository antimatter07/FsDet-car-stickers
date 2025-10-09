import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm



from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous, LastLevelMaxPool

class WeightedFPN(Backbone):
    """
    FPN with learnable fusion weights for each level:
    y = α * lateral + β * upsample(top-down)
    where α, β are learned scalars (per FPN level).
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None
    ):
        super().__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        _assert_strides_are_log2_contiguous(strides)

        use_bias = norm == ""
        lateral_convs = []
        output_convs = []
        fuse_weights = nn.ModuleList()

        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )

            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"fpn_lateral{stage}", lateral_conv)
            self.add_module(f"fpn_output{stage}", output_conv)

            # Create learnable weights for fusion (α, β)
            alpha = nn.Parameter(torch.ones(1))
            beta = nn.Parameter(torch.ones(1))
            fuse_weights.append(nn.ParameterList([alpha, beta]))

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Reverse order for top-down fusion
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.fuse_weights = fuse_weights[::-1]
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        self.top_block = top_block

        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {f: out_channels for f in self._out_features}
        self._size_divisibility = strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        for idx, (lateral_conv, output_conv, fuse_w) in enumerate(
            zip(self.lateral_convs, self.output_convs, self.fuse_weights)
        ):
            if idx > 0:
                lateral_name = self.in_features[-idx - 1]
                lateral_features = lateral_conv(bottom_up_features[lateral_name])
                top_down = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")

                alpha, beta = fuse_w
                fused = alpha * lateral_features + beta * top_down

                prev_features = fused
                results.insert(0, output_conv(fused))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_weighted_resnet_fpn_backbone(cfg, input_shape):
    """
    Build a ResNet-FPN backbone with learnable fusion weights.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = WeightedFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )
    return backbone
