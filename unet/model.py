import torch
from torch import nn
from torch.nn import functional as F

from config import Config


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(Config.GROUPS, out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(Config.GROUPS, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="nearest",
        )
        x = self.project(x)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class CorrectionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        w0, w1, w2, w3 = Config.WIDTHS

        self.enc1 = DoubleConv(Config.CHANNELS, w0)
        self.enc2 = DoubleConv(w0, w1)
        self.enc3 = DoubleConv(w1, w2)
        self.bottleneck = DoubleConv(w2, w3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = UpBlock(w3, w2)
        self.up2 = UpBlock(w2, w1)
        self.up3 = UpBlock(w1, w0)

        self.output = nn.Conv2d(
            w0,
            Config.CHANNELS,
            kernel_size=1,
            bias=True,
        )

    def forward(self, delta):
        e1 = self.enc1(delta)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        middle = self.bottleneck(self.pool(e3))

        x = self.up1(middle, e3)
        x = self.up2(x, e2)
        x = self.up3(x, e1)

        return self.output(x)