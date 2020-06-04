import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np



def compose_transformations(cfg):
    transformations = []

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
