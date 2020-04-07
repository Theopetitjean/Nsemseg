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
import glob

# -----------------------------------------------------------------------------#
#                           tentative d'adaptation
# -----------------------------------------------------------------------------#
from utils.misc import remap_mask


num_classes = 19
ignore_label = 255

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32, 56, 165, 134]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

# def make_dataset(im_folder, seg_folder, im_file_ending, seg_file_ending):
#     items = list()
#     for root, subdirs, files in os.walk(im_folder):
#         for ff in files:
#             if ff.endswith(im_file_ending):
#                 # Create file name for segmentation
#                 seg_path = os.path.join(seg_folder, root.replace(im_folder, '').strip('/'),
#                                         ff.replace(im_file_ending, seg_file_ending))
#                 # If segmentation exists, add it to list of files
#                 if os.path.isfile(seg_path):
#                     items.append((os.path.join(root, ff), seg_path))
#     return items

# -----------------------------------------------------------------------------#
#                          code original
# -----------------------------------------------------------------------------#
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
                 img_ext='.jpg',

#------------------------------------------------------------------------------#
                 id_to_trainid=None,
                 joint_transform=None,
                 sliding_crop=None,
                 transform=None,
                 target_transform=None,
                 transform_before_sliding=None
#------------------------------------------------------------------------------#
                 ):
        super(MonoDataset, self).__init__()
#------------------------------------------------------------------------------#
        # ata_path1 = "/media/HDD1/datasets/Creusot_Jan15/Creusot_3/*.jpg"
        # seg_folder = "media/HDD1/NsemSEG/Result_fold/"
        # im_file_ending = '.jpg'
        # seg_file_ending = 'jpg'
        # self.imgs = make_dataset(data_path1 , seg_folder, im_file_ending, seg_file_ending)
        # if len(self.imgs) == 0:
        #     raise RuntimeError('Found 0 images, please check the data set')
        items = glob.glob("/media/HDD1/datasets/Creusot_Jan15/Creusot_3/*.jpg", recursive=True)
#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.transform_before_sliding = transform_before_sliding
        self.id_to_trainid = id_to_trainid
#------------------------------------------------------------------------------#

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

        # self.load_depth = self.check_depth()

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

        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        if self.id_to_trainid is not None:
            mask = np.array(mask)
            mask_copy = mask.copy()
            for k, v in self.id_to_trainid.items():
                mask_copy[mask == k] = v
            mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            # from 0,1,2,...,255 to 0,1,2,3,... (to set introduced pixels due to transform to ignore)
            mask = remap_mask(mask, 0, ignore_label)
            img, mask = self.joint_transform(img, mask)
            mask = remap_mask(mask, 1, ignore_label)  # back again

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.sliding_crop is not None:
            if self.transform_before_sliding is not None:
                img = self.transform_before_sliding(img)
            img_slices, slices_info = self.sliding_crop(img)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            img = torch.stack(img_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, mask


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






# from sklearn.model_selection import train_test_split
# import sys
# from os import walk
# import os
# import glob
#
#
#
# assert(len(sys.argv) > 2)
#
#
#
# folder = sys.argv[1]
# split_name = sys.argv[2]
# folders = [x[0] for x in os.walk(folder)]
# folders.pop(0)
# images = []
# for f in folders:
#     fol = list(dict.fromkeys(glob.iglob(os.path.join(f, "*.jpg"))))
#     if 'rain' in f:
#         lentoremove = len(fol)
#     else:
#         lentoremove = len(fol) - 1
#
#
#
#     removed = os.path.join(f, str(lentoremove).zfill(5) + '.jpg')
#     fol.remove(removed)
#     if 'rain' in f:
#         removed = os.path.join(f, str(1).zfill(5) + '.jpg')
#         fol.remove(removed)
#     else:
#         removed = os.path.join(f, str(0).zfill(5) + '.jpg')
#         fol.remove(removed)
#
#
#
#     images += fol
#
#
#
#
# # for f in folders:
# #     images.append('test')
#
#
#
# # f = glob.glob(folder + '/**/*.jpg', recursive=True)
#
#
#
#
# splits = train_test_split(images, test_size=0.1, random_state=0)
# qualifier = ['train', 'val']
#
#
#
#
# for idx, lst in enumerate(splits):
#     fileout = f'{split_name}/{qualifier[idx]}_files.txt'
#     with open(fileout, 'w') as outfile:
#         for img in lst:
#             img_ = img.split('/')[-1]
#             index = str(int(img_.split('.')[0]))
#             if not int(index) == 0:
#                 firstpart = img.split('/')[-2]
#                 to_append = f'{firstpart} {index} l\n'
#                 outfile.write(to_append)
#
#
#
# # print(len(f))
# # print(len(out[0]))
# # print(len(out[1]))





    #def check_depth(self):
    #    raise NotImplementedError

    #def get_depth(self, folder, frame_index, side, do_flip):
    #    raise NotImplementedError


    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
