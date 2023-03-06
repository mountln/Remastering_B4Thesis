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
            padding=padding)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv3d(x)), inplace=False)


class SpatialConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3),
                 stride=1, padding=1):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv2d(x)), inplace=False)


class Upsample(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=(1, 2, 2)):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv3d = nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv3d(F.interpolate(
            x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False))), inplace=False)


class EfficientAttention(nn.Module):
    def __init__(self, in_channels_s, in_channels_r):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels_s, in_channels_s // 8, 1)
        self.key_conv = nn.Conv3d(in_channels_r, in_channels_r // 8, 1)
        self.value_conv = nn.Conv3d(in_channels_r, in_channels_r, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, source, reference):
        sB, sC, sT, sH, sW = source.size()
        rB, rC, rT, rH, rW = reference.size()
        key = self.key_conv(reference).view(rB, -1, rT * rH * rW)
        key = F.softmax(key, dim=2)
        value = self.value_conv(reference).view(rB, -1, rT * rH * rW)
        context = key @ value.transpose(1, 2)

        query = self.query_conv(source).view(sB, -1, sT * sH * sW)
        query = F.softmax(query, dim=1)
        attention = (context.transpose(1, 2) @ query).view(sB, -1, sT, sH, sW)

        out = self.gamma * attention + source

        return out


class NetworkR(nn.Module):
    def __init__(self, loop=2):
        super().__init__()

        self.loop = loop

        self.down_src = nn.Sequential(
            nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)),
            TempConv(1, 42, kernel_size=3, stride=(1, 2, 2), padding=0),
            TempConv(42, 127, kernel_size=3, padding=1),
            TempConv(127, 117, kernel_size=3, padding=1),
            TempConv(117, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            TempConv(256, 256, kernel_size=3, padding=1),
            TempConv(256, 256, kernel_size=3, padding=1),
        )

        self.down_ref = nn.Sequential(
            SpatialConv(1, 42, kernel_size=3, stride=2, padding=1),
            SpatialConv(42, 127, kernel_size=3, padding=1),
            SpatialConv(127, 117, kernel_size=3, padding=1),
            SpatialConv(117, 256, kernel_size=3, stride=2, padding=1),
            SpatialConv(256, 256, kernel_size=3, padding=1),
            SpatialConv(256, 256, kernel_size=3, padding=1),
        )

        self.up = nn.Sequential(
            TempConv(256, 256, kernel_size=3, padding=1),
            TempConv(256, 256, kernel_size=3, padding=1),
            Upsample(256, 99),
            TempConv(99, 64, kernel_size=3, padding=1),
            TempConv(64, 64, kernel_size=3, padding=1),
            Upsample(64, 16),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.attention = EfficientAttention(256, 256)

    def forward(self, x):
        for _ in range(self.loop):
            x = (x + torch.tanh(self.up(self.attention(
                self.down_src(x), self.down_ref(x[:, :, 0, :, :]).unsqueeze_(2))))).clamp(0, 1)
        return x
