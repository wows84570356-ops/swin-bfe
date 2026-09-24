
"""ResCBAM-UNet, the reference architecture, rebuilt layer by layer from the
authors' public implementation.

Source: https://github.com/trungdungtdct/Astro_CBAM_Unet
(Astro_CBAM_Unet.ipynb), extracted 2026-08-31. The class definitions are
unmodified; only this header and the imports were added, so that the
retrained comparison is against the published architecture itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, dropout_rate=0.3):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation=1, dropout_rate=0.3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation=1):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_c, out_c, dilation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

    def _pad_to_match(self, x, skip):
        """Pad x to match the spatial dimensions of skip."""
        if x.size(2) < skip.size(2):
            pad_h = skip.size(2) - x.size(2)
            pad_w = skip.size(3) - x.size(3)
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x


class ResCBAM_UNet_v4(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ResCBAM_UNet_v4, self).__init__()


        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)


        self.b1 = ResConvBlock(512, 640, dilation=2, dropout_rate=dropout_rate)
        self.b2 = ResConvBlock(640, 768, dilation=4, dropout_rate=dropout_rate)
        self.b3 = ResConvBlock(768, 896, dilation=8, dropout_rate=dropout_rate)
        self.b4 = ResConvBlock(896, 1024, dilation=16, dropout_rate=dropout_rate)


        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)


        self.final_conv = nn.Conv2d(64, 2, kernel_size=1, stride=1)

    def forward(self, inputs):

        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)


        b = self.b1(p4)
        b = self.b2(b)
        b = self.b3(b)
        b = self.b4(b)


        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        x = self.final_conv(d4)

        return x


class ResCBAMUNetWrapped(nn.Module):

    def __init__(self, dropout_rate=0.3, **kw):
        super().__init__()

        self.net = ResCBAM_UNet_v4(dropout_rate=dropout_rate)

    def forward(self, x, depth=None):
        y = self.net(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.shape[1] == 2:


            y = y[:, 1:2] - y[:, 0:1]
        z = torch.zeros_like(y)
        return {"logits": y, "width": F.softplus(z), "logvar": z}
