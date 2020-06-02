import os

import torchvision.transforms.functional as TF
from torchvision import transforms

import numpy as np
import torch
from scipy import ndimage
from unet.utils.utils import *


def compose_transformations(cfg):
    transformations = []

    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        transformations.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
        transformations.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        img, label = args
        img_tensor = TF.to_tensor(img)
        label_tensor = TF.to_tensor(label)
        return img_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, args):
        img, label = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)

        if vertical_flip:
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)

        img = img.copy()
        label = label.copy()

        return img, label


class RandomRotate(object):
    def __call__(self, args):
        img, label = args
        k = np.random.randint(1, 4) # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img, label


class ResizedRandomCrops(object):
    def __call__(self, args):
        img, label = args
        k = np.random.randint(1, 4) # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img, label


class VARI(object):
    def __call__(self, args):
        input, label, image_path = args
        image_name = os.path.basename(image_path)
        dir_name = os.path.dirname(image_path)
        vari_path = os.path.join(dir_name, 'clahe_vari' ,image_name)
        if os.path.exists(vari_path):
            mask = imread_cached(vari_path).astype(np.float32)[...,[0]]
            input_t = np.concatenate([input, mask], axis=-1)
            return input_t, label, image_path
        # Input is in BGR
        assert input.shape[1] == input.shape[2] and torch.is_tensor(input), 'invalid tensor, did you forget to put VARI after Np2Torch?'
        R = input[0]
        G = input[1]
        B = input[2]
        eps = 1e-6
        VARI = (G-R) / 2 * (eps + G+R-B) + 0.5 # Linearly transformed to be [0, 1]
        VARI = VARI.unsqueeze(0)
        input_t = torch.cat([input, VARI])
        return input_t, label, image_path


class UniformCrop(object):
    '''
    Performs uniform cropping on numpy images (cv2 images)
    '''
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def random_crop(self, input, label):
        image_size = input.shape[-2]
        crop_limit = image_size - self.crop_size
        x, y = np.random.randint(0, crop_limit, size=2)

        input = input[y:y+self.crop_size, x:x+self.crop_size, :]
        label = label[y:y+self.crop_size, x:x+self.crop_size]
        return input, label

    def __call__(self, args):
        input, label, image_path = args
        input, label = self.random_crop(input, label)
        return input, label, image_path


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args):
        input, label, image_path = args

        SAMPLE_SIZE = 5 # an arbitrary number that I came up with
        BALANCING_FACTOR = 200

        random_crops = [self.random_crop(input, label) for i in range(SAMPLE_SIZE)]
        # TODO Multi class vs edge mask
        weights = []
        for input, label in random_crops:
            if label.shape[2] >= 4:
                # Damage detection, multi class, excluding backround
                weights.append(label[...,:-1].sum())
            elif label.shape[2] > 1:
                # Edge Mask, excluding edge masks
                weights.append(label[...,0].sum())
            else:
                weights.append(label.sum())
        crop_weights = np.array([label.sum() for input, label in random_crops]) + BALANCING_FACTOR
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(SAMPLE_SIZE, p=crop_weights)
        input, label = random_crops[sample_idx]

        return input, label, image_path


class Resize(object):

    def __init__(self, scale, resize_label=True):
        self.scale = scale
        self.resize_label = resize_label

    def __call__(self, args):
        input, label, image_path = args

        input = cv2.resize(input, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.resize_label:
            label = cv2.resize(label, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        return input, label, image_path


class BGR2RGB():
    def __call__(self, args):
        input, label, image_path = args
        input = bgr2rgb(input)
        return input, label, image_path


def bgr2rgb(img):
    return img[..., [2, 1, 0]]