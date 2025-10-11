import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm



from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone import BACKBONE_REGISTRY, FPN
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


class RefineFPN(Backbone):
    """
    Standard FPN (additive fusion) with optional refinement conv blocks on low-level
    pyramid outputs (P2, P3). Keeps final output channel count identical across levels
    so it's fully compatible with RPN / ROI heads.
    """

    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, refine_levels=(2, 3)):
        """
        Args:
            bottom_up (Backbone): backbone producing features such as res2..res5
            in_features (list[str]): names of backbone features used by FPN (high->low res order)
            out_channels (int): number of channels in FPN outputs (kept the same across levels)
            norm (str): normalization string, passed to get_norm
            top_block (nn.Module or None): optional block to add p6/p7 etc.
            refine_levels (iterable): pyramid levels (integers) to apply refinement blocks to (e.g. (2,3))
        """
        super().__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, "in_features must be provided"

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        _assert_strides_are_log2_contiguous(strides)

        use_bias = norm == ""
        lateral_convs = []
        output_convs = []
        refinement_blocks = nn.ModuleDict()

        # Build lateral and output convs (1x1 lateral, 3x3 output) for each in_feature
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
            # register modules with names matching Detectron2's FPN naming
            self.add_module(f"fpn_lateral{stage}", lateral_conv)
            self.add_module(f"fpn_output{stage}", output_conv)

            # refinement blocks for selected low-level pyramid stages (p2, p3 by default)
            if stage in refine_levels:
                ref = nn.Sequential(
                    Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                )
                # initialize the convs inside refinement
                weight_init.c2_xavier_fill(ref[0])
                weight_init.c2_xavier_fill(ref[2])
                refinement_blocks[f"p{stage}"] = ref

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # place convs in top-down order (low-res -> high-res) for clearer forward logic
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.refinement_blocks = refinement_blocks
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        self.top_block = top_block

        # compute output strides (p2, p3, ...)
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}
        if self.top_block is not None:
            # `stage` variable above refers to last stage computed in the loop; that's fine here
            last_stage = int(math.log2(strides[-1]))
            for s in range(last_stage, last_stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            x (Tensor): image tensor (N,C,H,W) passed to the bottom-up network
        Returns:
            dict[str->Tensor]: mapping p2,p3,... to feature maps (all having out_channels)
        """
        bottom_up_features = self.bottom_up(x)
        results = []

        # start from the last backbone feature (e.g., res5) and build top-down
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # iterate remaining levels: combine lateral + upsample(prev)
        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx == 0:
                continue  # already processed the first (lowest-res) entry
            lateral_name = self.in_features[-idx - 1]
            stage = int(lateral_name[-1])  # 'res3' -> 3 etc.

            lateral_features = lateral_conv(bottom_up_features[lateral_name])
            top_down = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")

            # standard additive fusion (no learnable scalars)
            fused = lateral_features + top_down

            # refinement for selected low-level stages (keeps spatial resolution)
            key = f"p{stage}"
            if key in self.refinement_blocks:
                fused = self.refinement_blocks[key](fused)

            prev_features = fused
            results.insert(0, output_conv(fused))

        # top block (e.g., LastLevelMaxPool or LastLevelP6P7)
        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results), f"Expected {len(self._out_features)} outputs, got {len(results)}"
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

# -----------------------------
# 🔹 Small Object Feature Enhancement (SOFE)
# -----------------------------
class SOFEBlock(nn.Module):
    """Enhances small-object features using multi-scale convolution branches."""
    def __init__(self, in_channels, out_channels):
        super(SOFEBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

        for layer in [self.branch1, self.branch3, self.branch5, self.fuse]:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)
        return F.relu(self.fuse(out))


# -----------------------------
# 🔹 CBAM Attention Block
# -----------------------------
class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial Attention)."""
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        # Channel attention
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        # Spatial attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        weight_init.c2_xavier_fill(self.fc1)
        weight_init.c2_xavier_fill(self.fc2)
        weight_init.c2_xavier_fill(self.spatial)

    def forward(self, x):
        # Channel attention
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.max(x, dim=2, keepdim=True)[0]
        mx = torch.max(mx, dim=3, keepdim=True)[0]  # ✅ torch 1.7 safe alternative
        ch_attn = torch.sigmoid(self.fc2(F.relu(self.fc1(avg + mx))))
        x = x * ch_attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_attn = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * sp_attn


# -----------------------------
# 🔹 FPN with SOFE + CBAM
# -----------------------------
class SofeCBAMFPN(Backbone):
    """
    Detectron2-compatible FPN backbone with SOFE + CBAM on *all* lateral levels.
    Uses standard additive fusion (no α, β weights).
    """
    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None):
        super().__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, "in_features must be provided"

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        _assert_strides_are_log2_contiguous(strides)

        use_bias = norm == ""
        lateral_convs = []
        output_convs = []
        self.sofe_blocks = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList()

        # Build lateral/output convs + attach SOFE+CBAM to each
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(in_channels, out_channels, 1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(out_channels, out_channels, 3, 1, 1, bias=use_bias, norm=output_norm)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

            self.add_module(f"fpn_lateral{int(math.log2(strides[idx]))}", lateral_conv)
            self.add_module(f"fpn_output{int(math.log2(strides[idx]))}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

            # Add SOFE + CBAM per level
            self.sofe_blocks.append(SOFEBlock(out_channels, out_channels))
            self.cbam_blocks.append(CBAMBlock(out_channels))

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.sofe_blocks = self.sofe_blocks[::-1]
        self.cbam_blocks = self.cbam_blocks[::-1]

        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        self.top_block = top_block

        # Output metadata
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}
        if self.top_block is not None:
            last_stage = int(math.log2(strides[-1]))
            for s in range(last_stage, last_stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = None

        for idx, (lateral_conv, output_conv, sofe, cbam) in enumerate(
            zip(self.lateral_convs, self.output_convs, self.sofe_blocks, self.cbam_blocks)
        ):
            lateral_name = self.in_features[-idx - 1]
            lateral_features = lateral_conv(bottom_up_features[lateral_name])

            # SOFE + CBAM enhancement
            enhanced = cbam(sofe(lateral_features))

            if prev_features is not None:
                top_down = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                enhanced = enhanced + top_down  # standard additive fusion

            prev_features = enhanced
            results.insert(0, output_conv(enhanced))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, results[-1])
            results.extend(self.top_block(top_block_in_feature))

        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


# -----------------------------
# 🔹 Register Backbone in Detectron2
# -----------------------------
@BACKBONE_REGISTRY.register()
def build_resnet_sofe_cbam_fpn(cfg, input_shape):
    """
    Registerable name: "build_resnet_sofe_cbam_fpn"
    Add this under MODEL.BACKBONE.NAME in config.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = SofeCBAMFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )
    return backbone





@BACKBONE_REGISTRY.register()
def build_resnet_fpn_refine_backbone(cfg, input_shape: ShapeSpec):
    """
    Builds a ResNet-FPN backbone that includes refinement convs on low-level FPN outputs.
    Registered name: "build_resnet_fpn_refine_backbone"
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_block = LastLevelMaxPool()
    backbone = RefineFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        refine_levels=(2, 3),  # default: refine p2 & p3
    )
    return backbone

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
