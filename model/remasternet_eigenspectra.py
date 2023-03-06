import torch
import torch.nn as nn
import torch.nn.functional as F

eigenspectra = dict()


def calculate_eigenspectrum(feature_map: torch.Tensor) -> torch.Tensor:
    """Calculate the eigenspectrum of `feature_map` (shape: [B, C, T, H, W])."""
    b, c, t, h, w = feature_map.size()
    feature_map = feature_map.permute(0, 2, 3, 4, 1).reshape(b * t * h * w, c)
    covariance_matrix = feature_map.T @ feature_map / (b * t * h * w)
    eigenspectrum = torch.linalg.eigvals(covariance_matrix)
    return eigenspectrum


class TempConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1, 3, 3),
                 stride=(1, 1, 1), padding=(0, 1, 1), layer_id=None):
        super(TempConv, self).__init__()
        self.conv3d = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm3d(out_planes)

        self.layer_id = layer_id

    def forward(self, x):
        global eigenspectra
        device = x.device

        key = str(self.layer_id) + "_before"
        eigenspectrum = eigenspectra.get(key, torch.zeros(
            x.size(1))).to(device)
        eigenspectra[key] = (eigenspectrum + calculate_eigenspectrum(x)) / 2.

        after_conv3d = self.conv3d(x)
        key = str(self.layer_id) + "_after"
        eigenspectrum = eigenspectra.get(key, torch.zeros(
            after_conv3d.size(1))).to(device)
        eigenspectra[key] = (eigenspectrum + calculate_eigenspectrum(after_conv3d)) / 2.

        return F.elu(self.bn(after_conv3d), inplace=False)


class Upsample(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=(1, 2, 2), layer_id=None):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv3d = nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_planes)

        self.layer_id = layer_id

    def forward(self, x):
        global eigenspectra
        device = x.device

        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear',
                          align_corners=False)
        key = str(self.layer_id) + "_before"
        eigenspectrum = eigenspectra.get(key, torch.zeros(
            x.size(1))).to(device)
        eigenspectra[key] = (eigenspectrum + calculate_eigenspectrum(x)) / 2.

        after_conv3d = self.conv3d(x)
        key = str(self.layer_id) + "_after"
        eigenspectrum = eigenspectra.get(key, torch.zeros(
            after_conv3d.size(1))).to(device)
        eigenspectra[key] = (eigenspectrum + calculate_eigenspectrum(after_conv3d)) / 2.

        return F.elu(self.bn(after_conv3d), inplace=False)


def update_eigenspectrum_last(self, input, output):
    x = input[0]
    key = "12_before"
    eigenspectrum = eigenspectra.get(key, torch.zeros(
        x.size(1))).to(x.device)
    eigenspectra[key] = (eigenspectrum + calculate_eigenspectrum(x)) / 2.
    key = "12_after"
    eigenspectrum = eigenspectra.get(key, torch.zeros(
        output.size(1))).to(output.device)
    eigenspectra[key] = (eigenspectrum + calculate_eigenspectrum(output)) / 2.


class NetworkR(nn.Module):
    def __init__(self):
        super(NetworkR, self).__init__()

        self.layers = nn.Sequential(
            nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)),
            TempConv(1, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                     padding=(0, 0, 0), layer_id=0),
            TempConv(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=1),
            TempConv(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=2),
            TempConv(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                     padding=(1, 1, 1), layer_id=3),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=4),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=5),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=6),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=7),
            Upsample(256, 128, layer_id=8),
            TempConv(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=9),
            TempConv(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), layer_id=10),
            Upsample(64, 16, layer_id=11),
            nn.Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.layers[-1].register_forward_hook(update_eigenspectrum_last)

    def forward(self, x):
        # global eigenspectra

        # in1 = x.clone() - 0.4462414

        # out1 = self.layers(in1)
        # eigenspectrum = eigenspectra.get("12_before", torch.zeros(
        #     out1.size(1) * out1.size(2)))
        # eigenspectra["12_before"] = (eigenspectrum + calculate_eigenspectrum(out1)) / 2.

        # after_conv3d = self.conv3d(out1)
        # eigenspectrum = eigenspectra.get("12_after", torch.zeros(
        #     after_conv3d.size(1) * after_conv3d.size(2)))
        # eigenspectra["12_after"] = (eigenspectrum + calculate_eigenspectrum(after_conv3d)) / 2.

        return (x + torch.tanh(self.layers(x.clone() - 0.4462414))).clamp(0, 1)
