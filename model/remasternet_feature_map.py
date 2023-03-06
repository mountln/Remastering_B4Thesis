import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class TempConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1, 3, 3),
                 stride=(1, 1, 1), padding=(0, 1, 1), instance_id=None):
        super(TempConv, self).__init__()
        self.instance_id = instance_id
        self.conv3d = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        folder_top = os.path.join("feature_maps", self.instance_id)
        os.makedirs(folder_top, exist_ok=True)

        folder0 = os.path.join(folder_top, "0")
        os.makedirs(folder0, exist_ok=True)
        _, _, _, h, w = x.size()
        images = x.view(-1, 1, h, w).pow(2)
        utils.save_images(images, folder0)

        after_conv3d = self.conv3d(x)

        folder1 = os.path.join(folder_top, "1")
        os.makedirs(folder1, exist_ok=True)
        _, _, _, h, w = after_conv3d.size()
        utils.save_images(after_conv3d.view(-1, 1, h, w).pow(2), folder1)

        after_bn = self.bn(after_conv3d)

        folder2 = os.path.join(folder_top, "2")
        os.makedirs(folder2, exist_ok=True)
        _, _, _, h, w = after_bn.size()
        utils.save_images(after_bn.view(-1, 1, h, w).pow(2), folder2)

        after_activation = F.elu(after_bn, inplace=False)

        folder3 = os.path.join(folder_top, "3")
        os.makedirs(folder3, exist_ok=True)
        _, _, _, h, w = after_bn.size()
        utils.save_images(after_bn.view(-1, 1, h, w).pow(2), folder3)

        return after_activation


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


class UpsampleConcat(nn.Module):
    def __init__(self, in_planes_up, in_planes_flat, out_planes):
        super(UpsampleConcat, self).__init__()
        self.conv3d = TempConv(in_planes_up + in_planes_flat,
                               out_planes,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1))

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        return self.conv3d(x)


class NetworkR(nn.Module):
    def __init__(self):
        super(NetworkR, self).__init__()

        self.layers = nn.Sequential(
            nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)),
            TempConv(1, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                     padding=(0, 0, 0), instance_id="001"),
            TempConv(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="002"),
            TempConv(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="003"),
            TempConv(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                     padding=(1, 1, 1), instance_id="004"),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="004"),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="005"),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="006"),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="007"),
            Upsample(256, 128),
            TempConv(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="008"),
            TempConv(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), instance_id="009"),
            Upsample(64, 16),
            nn.Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1))
        )

    def forward(self, x):
        return (x + torch.tanh(self.layers(x.clone() - 0.4462414))).clamp(0, 1)
