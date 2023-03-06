import math
import os
import random
import shutil
import time

import torch
from torch.nn.functional import mse_loss, interpolate
from torchvision import models, utils
from torchvision.io import encode_jpeg, decode_jpeg

# from torchvision import transforms
# from torch.utils.data import Dataset
# from skimage import color
# import numpy as np
# from PIL import Image
# import cv2
# import os


class PerceptualLoss(torch.nn.Module):
    """Perceptual loss for a batch of Tensor images (shape: (B, C, H, W))."""
    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')):
        super(PerceptualLoss, self).__init__()
        pretrained_features = models.vgg16(pretrained=True).features
        blocks = (
            pretrained_features[0:4],
            pretrained_features[4:9],
            pretrained_features[9:16],
            pretrained_features[16:23],
            pretrained_features[23:30]
        )
        self.blocks = torch.nn.ModuleList(blocks).eval()

        block_names = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3')
        assert(len(blocks) == len(block_names))
        self.block_names = block_names

        # check if layers is valid
        for layer in layers:
            if layer not in block_names:
                raise ValueError(f"Layer '{layer}' is not in {block_names}.")
        self.layers = layers

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # If channel == 1, expand to 3.
        if input.size(-2) == 1:
            input = input.expand(-1, 3, -1, -1)
            target = target.expand(-1, 3, -1, -1)

        # Normalization
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Compute loss
        loss = 0.0
        for layer, block in zip(self.block_names, self.blocks):
            input = block(input)
            target = block(target)
            if layer in self.layers:
                loss += mse_loss(input, target)
        return loss


def add_gaussian_noise(input, mean=0.0, stddev=1.0):
    """Apply gaussian noise to the input."""
    return input + mean + torch.randn_like(input) * stddev


def compress_jpeg(input, quality_range):
    """
    Compress the input sequence using jpeg compression,
    The values of the input tensor are float between 0.0 and 1.0.
    """
    input_cpu = input.mul_(255.).clamp_(0, 255.).to(torch.device('cpu'), torch.uint8)
    outputs = []
    for frame in input_cpu:
        jpeg = encode_jpeg(frame, quality=int(random.uniform(*quality_range)))
        outputs.append(decode_jpeg(jpeg).to(input.device, torch.float).div_(255.))
    return torch.stack(outputs)


def random_bicubic_blur(input, scale_range):
    """Blur function using bicubic down-sampling."""
    height, width = input.size()[-2:]
    scale = random.uniform(*scale_range)
    output = interpolate(input, scale_factor=scale, mode='bicubic',
                         align_corners=False, recompute_scale_factor=True)
    output = interpolate(output, size=(height, width), mode='bicubic',
                         align_corners=False)
    return output


def save_frames(tensor, directory, filename_prefix='', format='png', start_num=0):
    num_frames = tensor.size(0)
    for i in range(num_frames):
        if filename_prefix == '':
            filename = str(i + start_num).zfill(5) + '.' + format
        else:
            filename = filename_prefix + '_' + str(i + start_num).zfill(
                len(str(num_frames))) + '.' + format
        file_path = os.path.join(directory, filename)
        utils.save_image(tensor[i, :, :, :], file_path)


def save_images(tensor, directory, filename=None, filename_prefix='',
                format='png', grid=True, start_num=1):
    os.makedirs(directory, exist_ok=True)
    if not filename:
        current_time = time.strftime("%Y%m%d_%H:%M:%S", time.localtime())
        filename = filename_prefix + '_' + current_time + '.' + format

    if grid:
        file_path = os.path.join(directory, filename)
        utils.save_image(list(tensor), file_path)
        print_log("=> images were saved to %s" % file_path)
    else:
        for i, img in enumerate(list(tensor), start=start_num):
            filename = filename_prefix + '%07d.' % i + format
            file_path = os.path.join(directory, filename)
        print_log("=> images were saved to %s" % directory)


def save_checkpoint(state, path='checkpoint.pth.tar', is_best=False):
    print_log("=> saving checkpoint '{}'".format(path))
    torch.save(state, path)
    print_log("=> checkpoint was saved")
    if is_best:
        directory, filename = os.path.split(path)
        best_checkpoint_path = os.path.join(directory, 'best_' + filename)
        shutil.copyfile(path, best_checkpoint_path)
        print_log("=> checkpoint was copied to '{}'".format(best_checkpoint_path))


def print_log(*args, end='\n'):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), end=': ')
    for string in args:
        print(string, end=" ")
    print(end=end)


def psnr(mse):
    return 20. * math.log10(1.0 / math.sqrt(mse))

# def convertLAB2RGB(lab):
#     lab[:, :, 0:1] = lab[:, :, 0:1] * 100   # [0, 1] -> [0, 100]
#     lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)  # [0, 1] -> [-128, 128]
#     rgb = color.lab2rgb(lab.astype(np.float64))
#     return rgb
#
#
# def convertRGB2LABTensor(rgb):
#     # RGB -> LAB L[0, 100] a[-127, 128] b[-128, 127]
#     lab = color.rgb2lab(np.asarray(rgb))
#     ab = np.clip(lab[:, :, 1:3] + 128, 0, 255)  # AB --> [0, 255]
#     ab = transforms.ToTensor()(ab) / 255.
#     L = lab[:, :, 0] * 2.55  # L --> [0, 255]
#     L = Image.fromarray(np.uint8(L))
#     L = transforms.ToTensor()(L)  # tensor [C, H, W]
#     return L, ab.float()
#
#
# def addMergin(img, target_w, target_h, background_color=(0, 0, 0)):
#     width, height = img.size
#     if width == target_w and height == target_h:
#         return img
#     scale = max(target_w, target_h) / max(width, height)
#     width = int(width * scale / 16.) * 16
#     height = int(height * scale / 16.) * 16
#     img = transforms.Resize((height, width), interpolation=Image.BICUBIC)(img)
#
#     xp = (target_w - width) // 2
#     yp = (target_h - height) // 2
#     result = Image.new(img.mode, (target_w, target_h), background_color)
#     result.paste(img, (xp, yp))
#     return result
#
#
# class RemasterDataset(Dataset):
#     """Remaster dataset."""
#     def __init__(
#             self,
#             root,
#             train=True,
#             sequence_size=5,
#             transform=None,
#             noise_k=(0.5, 1.0)
#     ):
#         """
#         Parameters
#         ----------
#             root: string
#                 Root directory of the dataset.
#             train: bool
#                 If True, creates dataset from train set,
#                 otherwise from test set.
#             sequence_size: int, optional
#                 Number of frames in a sequence.
#             transform: callable, optional
#                 Optional transform to be applied.
#             noise_k: Tuple(int, int), optional
#                 Range of coefficient k.
#         """
#         self.root = root
#         self.sequence_size = sequence_size
#         self.transform = transform
#         self.noise_k = noise_k
#
#         # Generate the list of frames directory
#         self._videos_dir = os.path.join(root, 'videos', 'train' if train else 'test')
#         self.frames_dir_list = [d for d in sorted(os.listdir(self._videos_dir))
#                                 if os.path.isdir(os.path.join(self._videos_dir, d))]
#         # Generate the list of all frames
#         self.frames_list = []
#         for i, frames_dir_name in enumerate(self.frames_dir_list):
#             frames_dir = os.path.join(self._videos_dir, frames_dir_name)
#             frames_list = [(i, frame_img_name) for frame_img_name in
#                            sorted(os.listdir(frames_dir))]
#             self.frames_list += frames_list
#
#         # Generate the lists of noise directory
#         noise_data_dir = os.path.join(root, 'noise_data')
#         self.noise_dir_list = [d for d in sorted(os.listdir(noise_data_dir))
#                                if os.path.isdir(os.path.join(noise_data_dir, d))]
#         # Generate the lists of all noise images
#         self.noise_list = []
#         for i, noise_dir_name in enumerate(self.noise_dir_list):
#             noise_dir = os.path.join(noise_data_dir, noise_dir_name)
#             noise_list = [(i, noise_img_name) for noise_img_name in
#                           sorted(os.listdir(noise_dir))]
#             self.noise_list += noise_list
#
#     def __len__(self):
#         return len(self.frames_list) - self.sequence_size
#
#     def __getitem__(self, idx):
#         # Ensure that all frames in the sequence are from the same video
#         video_id = self.frames_list[idx][0]
#         for i in range(1, self.sequence_size):
#             curr_video_id = self.frames_list[i + idx][0]
#             if curr_video_id != video_id:
#                 # If frames are not from the same video, then change idx to
#                 # get a suitable sequence
#                 idx -= self.sequence_size
#                 self.__getitem__(0 if idx < 0 else idx)
#
#         frames_dir = os.path.join(self._videos_dir,
#                                   self.frames_dir_list[video_id])
#         sequence_input, sequence_target = None, None
#         for i in range(idx, idx + self.sequence_size):
#             # Read frame
#             frame_file_name = self.frames_list[i][1]
#             frame_path = os.path.join(frames_dir, frame_file_name)
#             frame = cv2.imread(frame_path)
#             # Get the degraded frame
#             k = random.uniform(self.noise_k[0], self.noise_k[1])
#             k = -k if random.random() < 0.5 else k
#             degraded_frame = self._degrade(frame, k)
#             # Transform
#             if not self.transform:
#                 self.transform = transforms.ToTensor()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             degraded_frame = cv2.cvtColor(degraded_frame, cv2.COLOR_BGR2RGB)
#             frame = self.transform(Image.fromarray(frame))
#             degraded_frame = self.transform(Image.fromarray(degraded_frame))
#
#             # Change shape to (c, 1, h, w)
#             frame.unsqueeze_(1)
#             degraded_frame.unsqueeze_(1)
#             # Concatenate the sequence
#             sequence_input = degraded_frame if i == idx else \
#                 torch.cat((sequence_input, degraded_frame), -3)
#             sequence_target = frame if i == idx else \
#                 torch.cat((sequence_target, frame), -3)
#         return sequence_input, sequence_target
#
#     def _degrade(self, frame, k=1.0):
#         """Method for degrading `frame` with coefficient `k`."""
#         # Read and resize a random noise image
#         noise_class_id, noise_img_name = random.choice(self.noise_list)
#         noise_path = os.path.join(self.root, 'noise_data',
#                                   self.noise_dir_list[noise_class_id],
#                                   noise_img_name)
#         noise = cv2.imread(noise_path)
#         h, w = frame.shape[0:2]
#         noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_AREA)
#         # Blend frame and noise
#         degraded_frame = cv2.addWeighted(frame, 1, noise, k, 0)
#         return degraded_frame
