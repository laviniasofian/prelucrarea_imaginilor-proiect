import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNetMic(nn.Module):
    def __init__(self, n_channels=1, n_classes=3):
        super().__init__()

        self.encoder1 = DoubleConv(n_channels, 8)
        self.encoder2 = DoubleConv(8, 16)
        self.encoder3 = DoubleConv(16, 32)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(32, 64)

        self.decoder3 = DoubleConv(64 + 32, 32)
        self.decoder2 = DoubleConv(32 + 16, 16)
        self.decoder1 = DoubleConv(16 + 8, 8)

        self.output = nn.Conv2d(8, n_classes, kernel_size=1)

    def upsample_and_join(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        x = self.bottleneck(self.pool(e3))

        x = self.decoder3(self.upsample_and_join(x, e3))
        x = self.decoder2(self.upsample_and_join(x, e2))
        x = self.decoder1(self.upsample_and_join(x, e1))

        return self.output(x)