# Custom layers for the HDC-CNN model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_coord_channels(h, w, device):
    y = torch.linspace(-1, 1, h, device=device).view(-1, 1).expand(h, w)
    x = torch.linspace(-1, 1, w, device=device).view(1, -1).expand(h, w)
    return torch.stack([x, y], dim=0)


class CoordConv(nn.Module):
    """Appends normalized x,y coordinate channels to input."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        coords = make_coord_channels(H, W, x.device).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)


class WSConv2d(nn.Conv2d):
    """Weight-standardized conv2d."""
    def forward(self, x):
        w = self.weight
        w = (w - w.mean(dim=[1,2,3], keepdim=True)) / (w.std(dim=[1,2,3], keepdim=True) + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerScale(nn.Module):
    def __init__(self, dim, init=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        keep = 1 - self.p
        mask = keep + torch.rand((x.shape[0],) + (1,)*(x.ndim-1), device=x.device, dtype=x.dtype)
        return x / keep * mask.floor_()


class ECA(nn.Module):
    """Efficient channel attention."""
    def __init__(self, ch, gamma=2, b=1):
        super().__init__()
        k = int(abs(math.log2(ch) / gamma + b / gamma))
        k = k if k % 2 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


def calc_same_pad(w, k, d=1, s=1):
    eff_k = (k - 1) * d + 1
    total = max(0, (math.ceil(w / s) - 1) * s + eff_k - w)
    return total // 2, total - total // 2


class ReflectPad3x1(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.pad = dilation

    def forward(self, x):
        return F.pad(x, (0, 0, self.pad, self.pad), mode='reflect')


class AntiAliasedDownsample(nn.Module):
    """Blur + strided conv for anti-aliased downsampling."""
    def __init__(self, ch):
        super().__init__()
        blur = torch.tensor([1., 4., 6., 4., 1.]) / 16.0
        self.register_buffer('blur', blur.view(1, 1, 1, 5).repeat(ch, 1, 1, 1))
        self.ch = ch
        self.dw = WSConv2d(ch, ch, (1, 5), stride=(1, 2), padding=0, groups=ch, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        pl, pr = calc_same_pad(W, 5, 1, 2)
        x = F.pad(x, (pl, pr, 0, 0))
        x = F.conv2d(x, self.blur, stride=1, padding=0, groups=self.ch)
        x = F.pad(x, (pl, pr, 0, 0))
        return self.dw(x)


class SeparableConv2d(nn.Module):
    """Depthwise separable conv with dilation."""
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.dilation = dilation
        if dilation >= 5:
            self.conv1 = WSConv2d(ch, ch, (3, 1), padding=0, dilation=(dilation, 1), groups=ch, bias=False)
            self.pad = ReflectPad3x1(dilation)
            self.conv2 = WSConv2d(ch, ch, (1, 3), padding=(0, dilation), dilation=(1, dilation), groups=ch, bias=False)
            self.swapped = True
        else:
            self.conv1 = WSConv2d(ch, ch, (1, 3), padding=(0, dilation), dilation=(1, dilation), groups=ch, bias=False)
            self.conv2 = WSConv2d(ch, ch, (3, 1), padding=(dilation, 0), dilation=(dilation, 1), groups=ch, bias=False)
            self.swapped = False

    def forward(self, x):
        if self.swapped:
            x = self.pad(x)
        x = self.conv1(x)
        return self.conv2(x)


class HDCBlock(nn.Module):
    """Hybrid dilated convolution block."""
    def __init__(self, ch, dilations, ls_init=1e-5, drop=0.0, smooth=False, smooth_k=3):
        super().__init__()
        d1, d2 = dilations
        self.pre = WSConv2d(ch, ch, (1, smooth_k), padding=(0, smooth_k//2), groups=ch, bias=False) if smooth else nn.Identity()
        self.sep1 = SeparableConv2d(ch, d1)
        self.sep2 = SeparableConv2d(ch, d2)
        self.ls = LayerScale(ch, ls_init)
        self.dp = DropPath(drop)

    def forward(self, x):
        out = self.pre(x)
        out = self.sep1(out)
        out = self.sep2(out)
        return x + self.dp(self.ls(out))


class ThinBottleneck(nn.Module):
    def __init__(self, ch, reduction=2):
        super().__init__()
        mid = max(ch // reduction, 48)
        self.pw1 = nn.Conv2d(ch, mid, 1, bias=False)
        self.gn = nn.GroupNorm(min(8, mid), mid)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(mid, ch, 1, bias=False)

    def forward(self, x):
        return self.pw2(self.act(self.gn(self.pw1(x))))


class LargeKernelResidual(nn.Module):
    def __init__(self, ch, ls_init=1e-4, drop=0.0):
        super().__init__()
        self.dw = WSConv2d(ch, ch, (1, 63), padding=(0, 124), dilation=(1, 4), groups=ch, bias=False)
        self.gn = nn.GroupNorm(min(8, ch), ch)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        nn.init.constant_(self.gn.weight, 0.1)
        self.ls = LayerScale(ch, ls_init)
        self.dp = DropPath(drop)

    def forward(self, x):
        out = self.pw(self.gn(self.dw(x)))
        return x + self.dp(self.ls(out))


class ASPPLite(nn.Module):
    """Lightweight atrous spatial pyramid pooling."""
    def __init__(self, in_ch=96, out_ch=72, ls_init=5e-5, drop=0.0):
        super().__init__()
        q = in_ch // 4
        s = in_ch // 6
        cat_ch = (in_ch * 7) // 6

        self.b1_dw = WSConv2d(in_ch, in_ch, (1, 7), padding=(0, 3), groups=in_ch, bias=False)
        self.b1_pw = nn.Conv2d(in_ch, q, 1, bias=False)
        self.b2_dw = WSConv2d(in_ch, in_ch, (1, 15), padding=(0, 14), dilation=2, groups=in_ch, bias=False)
        self.b2_pw = nn.Conv2d(in_ch, q, 1, bias=False)
        self.b3_dw = WSConv2d(in_ch, in_ch, (1, 21), padding=(0, 20), dilation=2, groups=in_ch, bias=False)
        self.b3_pw = nn.Conv2d(in_ch, q, 1, bias=False)
        self.b4 = nn.Conv2d(in_ch, s, 1, bias=False)
        self.h_dw = WSConv2d(in_ch, in_ch, (63, 1), padding=(31, 0), groups=in_ch, bias=False)
        self.h_pw = nn.Conv2d(in_ch, q, 1, bias=False)

        self.gn = nn.GroupNorm(min(8, cat_ch), cat_ch)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(cat_ch, out_ch, 1, bias=False)
        self.eca = ECA(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.ls = LayerScale(out_ch, ls_init)
        self.dp = DropPath(drop)

    def forward(self, x):
        h1 = self.h_pw(self.h_dw(x))
        b1 = self.b1_pw(self.b1_dw(x))
        b2 = self.b2_pw(self.b2_dw(x))
        b3 = self.b3_pw(self.b3_dw(x))
        b4 = self.b4(x)

        out = torch.cat([b1, b2, b3, b4, h1], dim=1)
        out = self.proj(self.act(self.gn(out)))
        out = self.eca(out)
        return self.shortcut(x) + self.dp(self.ls(out))
