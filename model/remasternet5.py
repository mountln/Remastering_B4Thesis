import torch
import torch.nn as nn
import torch.nn.functional as F


class TempConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1, 3, 3),
                 stride=1, padding=1):
        super(TempConv, self).__init__()
        self.conv3d = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode='reflect')
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv3d(x)), inplace=False)


class Upsample(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=(1, 2, 2)):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv3d = nn.Conv3d(in_planes, out_planes, kernel_size=3,
                                stride=1, padding=1, padding_mode='reflect')
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv3d(F.interpolate(
            x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False))), inplace=False)


class EfficientSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, t, h, w = x.size()
        key = self.key_conv(x).view(b, -1, t * h * w)
        key = F.softmax(key, dim=2)
        value = self.value_conv(x).view(b, -1, t * h * w)
        context = key @ value.transpose(1, 2)

        query = self.query_conv(x).view(b, -1, t * h * w)
        query = F.softmax(query, dim=1)
        attention = (context.transpose(1, 2) @ query).view(b, -1, t, h, w)

        out = self.gamma * attention + x

        return out


class NetworkR(nn.Module):
    def __init__(self, train=True, loop=3):
        super().__init__()

        if train:
            self.loop = 1
        else:
            self.loop = loop

        self.down = nn.Sequential(
            TempConv(1, 12, kernel_size=3, stride=(1, 2, 2), padding=1),
            TempConv(12, 35, kernel_size=3, padding=1),
            TempConv(35, 55, kernel_size=3, padding=1),
            TempConv(55, 167, kernel_size=3, stride=(1, 2, 2), padding=1),
            TempConv(167, 196, kernel_size=3, padding=1),
            TempConv(196, 187, kernel_size=3, padding=1),
        )

        self.attention = EfficientSelfAttention(187)

        self.up = nn.Sequential(
            TempConv(187, 103, kernel_size=3, padding=1),
            TempConv(103, 41, kernel_size=3, padding=1),
            Upsample(41, 23),
            TempConv(23, 21, kernel_size=3, padding=1),
            TempConv(21, 20, kernel_size=3, padding=1),
            Upsample(20, 15),
            nn.Conv3d(15, 1, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        for _ in range(self.loop):
            x = (x + torch.tanh(self.up(self.attention(self.down(x))))).clamp(0, 1)
        return x
