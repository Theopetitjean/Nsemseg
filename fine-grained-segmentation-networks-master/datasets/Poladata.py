# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import sys
import os
import random
import numpy as np
import copy
from PIL import Image, ImageFile  # using pillow-simd for increased speed
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms


# def pil_loader(path):
#     # open path as file to avoid ResourceWarning
#     # (https://github.com/python-pillow/Pillow/issues/835)
#     # with open(path, 'rb') as f:
#     #     with Image.open(f) as img:
#     #
#     #         return img.convert('RGB')
#
#     f = cv2.imread(path)
#     img = Image.fromarray(f, mode='RGB')
#
#     return img


def pil_loader(path):
    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # print(np.array(img.convert('RGB')))
            # exit(0)
            I0_45_90_135 = (0, 1, 2, 3)
            I45_0_135_90 = (1, 0, 3, 2)
            I135_90_45_0 = (3, 2, 1, 0)
            I90_135_0_45 = (2, 3, 0, 1)
            I45_90_135_0 = (1, 2, 3, 0)
            I135_0_45_90 = (3, 0, 1, 2)

            raw = np.array(img)
            images = np.array([raw[0::2, 0::2],  # 0
                               raw[0::2, 1::2],  # 1
                               raw[1::2, 1::2],  # 2
                               raw[1::2, 0::2]])  # 3
            images = images[I135_0_45_90, :, :]
            mat = np.array([[0.5, 0.5, 0.5, 0.5],
                            [1.0, 0.0, -1., 0.0],
                            [0.0, 1.0, 0.0, -1.]])

            stokes = np.tensordot(mat, images, 1)

            inte = stokes[0] / 2
            aop = np.mod(np.arctan2(stokes[2], stokes[1]) / 2., np.pi)
            dop = np.divide(np.sqrt(stokes[2]**2, stokes[1]**2),
                            stokes[0] * 2, out=np.zeros_like(stokes[0]),
                            where=stokes[0] != 0)
            inte = np.uint8(inte)

            aop = np.uint8((aop / np.pi) * 255)
            np.set_printoptions(threshold=sys.maxsize)
            # print(np.amax(dop), np.amin(dop))
            # import scipy.ndimage as ndimage
            # dop = ndimage.gaussian_filter(dop, sigma=3)

            # dop[dop <= 0.4] = 0
            # dop[dop > 0.4] = .8 * 255
            low_thresh = 0.4
            high_thresh = 0.4
            dop *= 255
            dop[dop < (low_thresh * 255)] = 0
            # dop[dop >= (high_thresh * 255)] = .8 * 255
            dop = np.uint8(dop)
            # print(np.amax(dop), np.amin(dop), np.mean(dop))
            # dop = dop > 0.5
            #
            # dop = np.uint8(dop * 255)
            # to_tensor = transforms.ToTensor()
            # dop = to_tensor(Image.fromarray(dop))
            # print(torch.max(dop))
            # exit(0)
            # print(np.amax(dop), np.amin(dop))

            # print(dop)
            # exit(0)

            # print(np.histogram(dop, bins='auto'))
            # print(dop)
            # print('inte: ', np.amax(inte), np.amin(inte))
            # print('aop: ', np.amax(aop), np.amin(aop))
            # print('dop: ', np.amax(dop), np.amin(dop))

            image = np.dstack((inte, aop, dop))

            # image = np.zeros((stokes.shape[1], stokes.shape[2], 3))
            #
            # for x in range(image.shape[2]):
            #     image[:, :, x] = inte
            # image = np.uint8(image)

            return Image.fromarray(image).convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=Image.NEAREST)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        # do_color_aug = False
        # do_flip = self.is_train and random.random() > 0.5
        do_flip = False

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder,
                                                          frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder,
                                                          frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # if self.load_depth:
        #     depth_gt = self.get_depth(folder, frame_index, side, do_flip)
        #     inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        #     inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
