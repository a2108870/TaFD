# -*- coding: utf-8 -*-
"""
models/standard_models.py
-------------------------
Standard ResNet-34 and MobileViT baseline models (no FCConv / routing / BPDA).
Input normalization is consistent with encoder.py.

Usage:
  from models.standard_models import create_standard_model
  model = create_standard_model(backbone='resnet', dataset='CIFAR100', num_classes=100)
"""

import math
import torch
import torch.nn as nn

try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    rearrange = None

# Dataset normalization statistics (consistent with encoder.py)
DATASET_STATS = {
    'CIFAR10':  {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'CIFAR100': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
}


class InputNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _get_normalize(dataset):
    stats = DATASET_STATS.get(dataset, DATASET_STATS['CIFAR100'])
    return InputNormalize(mean=stats['mean'], std=stats['std'])


# Standard ResNet-34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class StandardResNet34(nn.Module):
    """Standard ResNet-34, structurally aligned with TaFD ResNetEncoder (without FCConv/routing)."""

    def __init__(self, num_classes=100, dataset='CIFAR100'):
        super().__init__()
        self.input_normalize = _get_normalize(dataset)
        self.in_channels = 64

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        layers = [3, 4, 6, 3]
        self.layer1 = self._make_layer(64,  layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_normalize(x)
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        logits = self.fc(x)
        # Return tuple for compatibility with torchattacks model(x)[0] calls
        return (logits,)


# Standard MobileViT
def _conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup), nn.SiLU(),
    )

def _conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup), nn.SiLU(),
    )

class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.fn = fn
    def forward(self, x, **kw):
        return self.fn(self.norm(x), **kw)

class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class _Attention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=32, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads; self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = torch.matmul(self.attend(dots), v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                _PreNorm(dim, _Attention(dim, heads, dim_head, dropout)),
                _PreNorm(dim, _FeedForward(dim, mlp_dim, dropout)),
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x; x = ff(x) + x
        return x

class _MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res = (stride == 1 and inp == oup)
        self.pw1 = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        out = self.pw2(self.dw(self.pw1(x)))
        return (x + out) if self.use_res else out


class _MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.0):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = _conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = _conv_1x1_bn(channel, dim)
        self.transformer = _Transformer(dim, depth, 1, 32, mlp_dim, dropout)
        self.conv3 = _conv_1x1_bn(dim, channel)
        self.conv4 = _conv_nxn_bn(2 * channel, channel, kernel_size)
    def forward(self, x):
        y = x.clone()
        x = self.conv2(self.conv1(x))
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        return self.conv4(torch.cat((x, y), 1))


class StandardMobileViT(nn.Module):
    """Standard MobileViT, structurally aligned with TaFD MobileViTEncoder (without FCConv/routing)."""

    def __init__(self, num_classes=100, dataset='CIFAR100',
                 size=32, expansion=3, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        if not EINOPS_AVAILABLE:
            raise ImportError("StandardMobileViT requires 'einops'.")
        ih = iw = size
        self.input_normalize = _get_normalize(dataset)

        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        L = [2, 4, 3]

        self.conv1 = _conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([
            _MV2Block(channels[0], channels[1], 1, expansion),
            _MV2Block(channels[1], channels[2], 1, expansion),
            _MV2Block(channels[2], channels[3], 1, expansion),
            _MV2Block(channels[2], channels[3], 1, expansion),
            _MV2Block(channels[3], channels[4], 2, expansion),
            _MV2Block(channels[5], channels[6], 2, expansion),
            _MV2Block(channels[7], channels[8], 1, expansion),
        ])

        self.mvit = nn.ModuleList([
            _MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)),
            _MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)),
            _MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)),
        ])

        self.conv2 = _conv_1x1_bn(channels[-2], channels[-1])
        self.pool = nn.AvgPool2d(ih // 8, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        x = self.input_normalize(x)
        x = self.conv1(x)
        for i, blk in enumerate(self.mv2):
            x = blk(x)
            if i == 4: x = self.mvit[0](x)
            if i == 5: x = self.mvit[1](x)
            if i == 6: x = self.mvit[2](x)
        x = self.conv2(x)
        x = self.pool(x).view(-1, x.shape[1])
        logits = self.fc(x)
        return (logits,)


# Factory function
def create_standard_model(backbone='resnet', dataset='CIFAR100', num_classes=100):
    if backbone == 'resnet':
        return StandardResNet34(num_classes=num_classes, dataset=dataset)
    elif backbone == 'mobilevit':
        return StandardMobileViT(num_classes=num_classes, dataset=dataset)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
