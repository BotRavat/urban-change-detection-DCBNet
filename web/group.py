"""
model_arch.py — Fixed Model Architecture
=========================================
Fixes vs original:
1. ChangeDetectionNet.__init__: dropout_p is now a parameter (not a global)
2. ChangeDetectionNet.forward_with_stage2_features: _channel_exchange added back
3. ChangeRefinementNet: uses full CRFRefinementASPP refine head (not bare delta_head)
   so it matches the saved checkpoint keys exactly.

This version also adds simple binary change-detection metrics that can be reused
from other modules (e.g. Gradio app).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ══════════════════════════════════════════════════════════════
# STAGE 1 BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, ch, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.se(self.conv(x))


class DiffGate(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 8, ch, 1, bias=True),
            nn.Sigmoid(),
        )
        # Encourage gate to be relatively open at init
        nn.init.constant_(self.gate[3].bias, 3.0)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        g = self.gate(torch.abs(f1 - f2))
        return f1 * g, f2 * g


# ══════════════════════════════════════════════════════════════
# STAGE 1 — ChangeDetectionNet
# ══════════════════════════════════════════════════════════════

class ChangeDetectionNet(nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        """Stage‑1 change detector (ResNet‑50 backbone)."""
        super().__init__()
        backbone = models.resnet50(weights=None)  # weights=None avoids re-download

        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stage1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

        self.proj4 = nn.Conv2d(2048, 512, 1, bias=False)
        self.proj3 = nn.Conv2d(1024, 256, 1, bias=False)

        self.diff_gate2 = DiffGate(512)
        self.diff_gate3 = DiffGate(1024)

        self.cross_attn4 = nn.MultiheadAttention(512, 8, batch_first=True, dropout=0.1)
        self.cross_attn3 = nn.MultiheadAttention(256, 8, batch_first=True, dropout=0.1)

        self.dec4 = DecoderBlock(512, 256, skip_ch=256 * 3)
        self.dec3 = DecoderBlock(256, 128, skip_ch=512 * 3)
        self.dec2 = DecoderBlock(128, 64, skip_ch=256 * 3)
        self.dec1 = DecoderBlock(64, 64, skip_ch=64 * 3)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(32, 1, 1),
        )

        self.aux_dec4 = nn.Conv2d(256, 1, 1)
        self.aux_dec3 = nn.Conv2d(128, 1, 1)

    # ── helpers ──────────────────────────────────────────────

    def encode_partial(self, x: torch.Tensor):
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s0, s1, s2, s3

    @staticmethod
    def _rich_skip(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, torch.abs(a - b)], dim=1)

    @staticmethod
    def _cross_attn(mod: nn.MultiheadAttention, f1: torch.Tensor, f2: torch.Tensor):
        B, C, H, W = f1.shape
        q = f1.flatten(2).permute(0, 2, 1)  # [B,HW,C]
        k = f2.flatten(2).permute(0, 2, 1)
        out, _ = mod(q, k, k)
        return out.permute(0, 2, 1).reshape(B, C, H, W)

    @staticmethod
    def _channel_exchange(f1: torch.Tensor, f2: torch.Tensor):
        # Odd channels are swapped across the pair (as in original training code)
        B, C, H, W = f1.shape
        mask = torch.zeros(C, device=f1.device)
        mask[1::2] = 1.0
        mask = mask.view(1, C, 1, 1)
        return f1 * (1 - mask) + f2 * mask, f2 * (1 - mask) + f1 * mask

    # ── standard forward ─────────────────────────────────────

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        f1 = list(self.encode_partial(x1))
        f2 = list(self.encode_partial(x2))

        f1[3], f2[3] = self._channel_exchange(f1[3], f2[3])
        s4_1 = self.stage4(f1[3])
        s4_2 = self.stage4(f2[3])
        f1.append(s4_1)
        f2.append(s4_2)

        gf1_2, gf2_2 = self.diff_gate2(f1[2], f2[2])
        gf1_3, gf2_3 = self.diff_gate3(f1[3], f2[3])

        p1_4 = self.proj4(f1[4])
        p2_4 = self.proj4(f2[4])
        p1_3 = self.proj3(gf1_3)
        p2_3 = self.proj3(gf2_3)

        attn4 = self._cross_attn(self.cross_attn4, p1_4, p2_4)
        diff4 = torch.abs(p1_4 - attn4)

        attn3 = self._cross_attn(self.cross_attn3, p1_3, p2_3)
        diff3 = torch.cat([p1_3, attn3, torch.abs(p1_3 - attn3)], dim=1)

        d4 = self.dec4(diff4, diff3)
        d3 = self.dec3(d4, self._rich_skip(gf1_2, gf2_2))
        d2 = self.dec2(d3, self._rich_skip(f1[1], f2[1]))
        d1 = self.dec1(d2, self._rich_skip(f1[0], f2[0]))
        p1 = self.head(d1)

        return p1, self.aux_dec4(d4), self.aux_dec3(d3)

    # ── Stage‑2‑facing forward (exposes intermediate features) ─

    def forward_with_stage2_features(self, x1: torch.Tensor, x2: torch.Tensor):
        f1 = list(self.encode_partial(x1))
        f2 = list(self.encode_partial(x2))

        # channel exchange before stage4
        f1[3], f2[3] = self._channel_exchange(f1[3], f2[3])

        s4_1 = self.stage4(f1[3])
        s4_2 = self.stage4(f2[3])
        f1.append(s4_1)
        f2.append(s4_2)

        gf1_2, gf2_2 = self.diff_gate2(f1[2], f2[2])
        gf1_3, gf2_3 = self.diff_gate3(f1[3], f2[3])

        p1_4 = self.proj4(f1[4])
        p2_4 = self.proj4(f2[4])
        p1_3 = self.proj3(gf1_3)
        p2_3 = self.proj3(gf2_3)

        attn4 = self._cross_attn(self.cross_attn4, p1_4, p2_4)
        diff4 = torch.abs(p1_4 - attn4)

        attn3 = self._cross_attn(self.cross_attn3, p1_3, p2_3)
        diff3_attn = torch.abs(p1_3 - attn3)
        diff3_dec = torch.cat([p1_3, attn3, diff3_attn], dim=1)

        d4 = self.dec4(diff4, diff3_dec)
        d3 = self.dec3(d4, self._rich_skip(gf1_2, gf2_2))
        d2 = self.dec2(d3, self._rich_skip(f1[1], f2[1]))
        d1 = self.dec1(d2, self._rich_skip(f1[0], f2[0]))
        p1 = self.head(d1)

        diff2_raw = torch.abs(f1[2] - f2[2])  # [B,512,32,32]
        return p1, diff4, diff3_attn, diff2_raw, d1


# ══════════════════════════════════════════════════════════════
# STAGE 2 BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class RegionSegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_diff2 = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.region_conv1 = nn.Sequential(
            nn.Conv2d(257, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.region_conv2 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.region_out = nn.Conv2d(64, 1, 1)

    def forward(
        self,
        diff3_attn: torch.Tensor,
        diff2_raw: torch.Tensor,
        p1_sig: torch.Tensor,
    ) -> torch.Tensor:
        diff2 = self.proj_diff2(diff2_raw)
        p1_16 = F.interpolate(
            p1_sig,
            size=diff3_attn.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.region_conv1(torch.cat([diff3_attn, p1_16], dim=1))
        x = F.interpolate(
            x, size=diff2.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.region_conv2(torch.cat([x, diff2], dim=1))
        return self.region_out(x)


class ObjectAggregationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg_proj = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.ctx_proj = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.agg_fuse = nn.Sequential(
            nn.Conv2d(384, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, diff4: torch.Tensor, diff2_raw: torch.Tensor, region_map: torch.Tensor
    ) -> torch.Tensor:
        feat = self.agg_proj(diff2_raw)
        ctx = F.interpolate(
            self.ctx_proj(diff4),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        r16 = F.interpolate(
            torch.sigmoid(region_map),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        w = r16.sum(dim=(2, 3), keepdim=True) + 1e-6
        proto = (feat * r16).sum(dim=(2, 3), keepdim=True) / w
        return self.agg_fuse(torch.cat([feat, ctx, proto.expand_as(feat)], dim=1))


class CRFRefinementASPP(nn.Module):
    """ASPP‑style refinement used in ChangeRefinementNet."""

    def __init__(self, dilations=(1, 3, 6)):
        super().__init__()
        self.proj3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.region_proj = nn.Conv2d(1, 64, 1, bias=False)
        self.fuse_region = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        d1, d2, d3 = dilations

        def branch(d):
            return nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

        self.aspp_d1 = branch(d1)
        self.aspp_d2 = branch(d2)
        self.aspp_d3 = branch(d3)
        self.aspp_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, obj_feat: torch.Tensor, region_map: torch.Tensor) -> torch.Tensor:
        x = self.proj3(obj_feat)
        rg = self.region_proj(torch.sigmoid(region_map))
        x = self.fuse_region(torch.cat([x, rg], dim=1))

        x = self.up2(x)

        a1 = self.aspp_d1(x)
        a2 = self.aspp_d2(x)
        a3 = self.aspp_d3(x)
        gap = self.aspp_gap(x).expand_as(a1)
        x = self.aspp_fuse(torch.cat([a1, a2, a3, gap], dim=1))

        x = self.up1(x)
        x = self.up0(x)
        return x


# ══════════════════════════════════════════════════════════════
# STAGE 2 — ChangeRefinementNet
# ══════════════════════════════════════════════════════════════

class ChangeRefinementNet(nn.Module):
    def __init__(self, aspp_dilations=(1, 3, 6), delta_scale_init: float = 0.5):
        super().__init__()
        self.region_head = RegionSegmentationHead()
        self.object_agg = ObjectAggregationModule()
        self.refine = CRFRefinementASPP(aspp_dilations)
        self.delta_head = nn.Conv2d(16, 1, 1)  # input=16 matches refine output

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.delta_scale = nn.Parameter(torch.tensor(delta_scale_init))

    def forward(
        self,
        diff3_attn: torch.Tensor,
        diff4: torch.Tensor,
        diff2_raw: torch.Tensor,
        p1_logit: torch.Tensor,
    ):
        p1_sig = torch.sigmoid(p1_logit)
        gate = torch.clamp(4.0 * p1_sig * (1.0 - p1_sig), min=0.2)

        region_map = self.region_head(diff3_attn, diff2_raw, p1_sig)
        obj_feat = self.object_agg(diff4, diff2_raw, region_map)
        refined = self.refine(obj_feat, region_map)

        delta = (
            torch.tanh(self.delta_head(refined))
            * torch.clamp(self.delta_scale, 0.05, 2.0)
        )
        p2_logit = p1_logit + gate * delta
        return p2_logit, region_map, gate


# ══════════════════════════════════════════════════════════════
# METRICS HELPERS (for inference / evaluation)
# ══════════════════════════════════════════════════════════════

def confusion_from_masks(pred01, gt01):
    """
    Compute TP, TN, FP, FN from binary masks (0/1 or bool).

    Args:
        pred01: prediction mask, any array‑like of {0,1}
        gt01: ground‑truth mask, same shape as pred01

    Returns:
        (tp, tn, fp, fn) as Python ints.
    """
    p = np.asarray(pred01).astype(np.uint8)
    g = np.asarray(gt01).astype(np.uint8)
    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape}, gt {g.shape}")

    tp = int(((p == 1) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    return tp, tn, fp, fn


def compute_change_metrics(pred01, gt01, eps: float = 1e-8):
    """
    Compute standard binary segmentation metrics.

    Returns a dict with float metrics (not formatted strings):
        accuracy, precision, recall, f1, iou, tp, fp, fn
    """
    tp, tn, fp, fn = confusion_from_masks(pred01, gt01)

    denom_all = tp + tn + fp + fn + eps
    acc = (tp + tn) / denom_all
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }