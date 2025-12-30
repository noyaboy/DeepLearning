# Hybrid dilated CNN for cosmological parameter inference with uncertainty.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .base import BaseModel
from .registry import ModelRegistry
from .layers import (
    CoordConv, WSConv2d, LayerScale, DropPath, ECA,
    AntiAliasedDownsample, HDCBlock, ThinBottleneck,
    LargeKernelResidual, ASPPLite
)


@ModelRegistry.register("hdc_cnn")
class HDCCNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        dp = [i * 0.01 for i in range(6)]  # drop path rates
        self.use_checkpointing = False

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        self.coord_conv = CoordConv()

        # stage A
        self.downsample_a = AntiAliasedDownsample(34)
        self.down_a_pw = nn.Sequential(
            nn.Conv2d(34, 48, 1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU()
        )
        self.down_a_conv = nn.Conv2d(48, 48, 3, padding=1, bias=False)
        self.down_a_block = HDCBlock(48, (1, 2), 1e-5, dp[0], False)

        # stage B
        self.downsample_b = AntiAliasedDownsample(48)
        self.down_b_pw = nn.Sequential(
            nn.Conv2d(48, 96, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU()
        )
        self.down_b_conv = nn.Conv2d(96, 96, 3, padding=1, bias=False)
        self.down_b_block = HDCBlock(96, (2, 5), 2e-5, dp[1], True, 3)
        self.down_b_bottleneck = ThinBottleneck(96, 2)

        # stage C
        self.down_c_pre_dw = WSConv2d(96, 96, (1, 7), padding=(0, 3), groups=96, bias=False)
        self.down_c_pre_pw = nn.Sequential(
            nn.Conv2d(96, 96, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU()
        )
        self.down_c_pre_layerscale = LayerScale(96, 3e-5)
        self.down_c_pre_droppath = DropPath(dp[2])
        self.down_c_block1 = HDCBlock(96, (2, 5), 3e-5, dp[2], True, 3)
        self.down_c_block2 = HDCBlock(96, (3, 7), 3e-5, dp[3], True, 5)
        self.down_c_eca = ECA(96)

        # large kernel + aspp
        self.large_kernel = LargeKernelResidual(96, 1e-4, dp[4])
        self.large_kernel_bottleneck = ThinBottleneck(96, 2)
        self.aspp = ASPPLite(96, 72, 5e-5, dp[5])

        # neck and heads
        self.neck = nn.Sequential(
            nn.Conv2d(72, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mean_head = nn.Linear(64, 2)
        self.cov_head = nn.Linear(64, 3)

    def _enable_checkpointing(self):
        self.use_checkpointing = True

    def _ckpt(self, fn, x):
        return checkpoint(fn, x, use_reentrant=False) if self.use_checkpointing else fn(x)

    def forward(self, batch):
        x = batch['image']

        # encoder
        x = self.stem(x)
        x = self.coord_conv(x)

        x = self.downsample_a(x)
        x = self.down_a_pw(x)
        x = self.down_a_conv(x)
        x = self._ckpt(self.down_a_block, x)

        x = self.downsample_b(x)
        x = self.down_b_pw(x)
        x = self.down_b_conv(x)
        x = self._ckpt(self.down_b_block, x)
        x = self._ckpt(self.down_b_bottleneck, x)

        # stage C pre-block
        identity = x
        out = self.down_c_pre_dw(x)
        out = self.down_c_pre_pw(out)
        out = self.down_c_pre_layerscale(out)
        out = self.down_c_pre_droppath(out)
        x = out + identity

        x = self._ckpt(self.down_c_block1, x)
        x = self._ckpt(self.down_c_block2, x)
        x = self.down_c_eca(x)

        x = self._ckpt(self.large_kernel, x)
        x = self._ckpt(self.large_kernel_bottleneck, x)
        x = self._ckpt(self.aspp, x)
        x = self._ckpt(self.neck, x)

        x = self.gap(x).flatten(1)

        # heads (fp32 for numerical stability)
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            mean = self.mean_head(x)

            cov = self.cov_head(x)
            a, b, c = cov[:, 0], cov[:, 1], cov[:, 2]

            eps = 1e-6
            L11 = F.softplus(a) + eps
            L22 = F.softplus(c) + eps

            var1 = L11 * L11 + eps
            var2 = b * b + L22 * L22 + eps

            log_var = torch.stack([torch.log(var1), torch.log(var2)], dim=1)

        return {'mean': mean, 'log_var': log_var}
