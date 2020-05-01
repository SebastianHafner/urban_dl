#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import torch
import json
from pathlib import Path
from unet.utils import *
from torchvision import transforms
from unet.augmentations import *
from preprocessing.utils import *
from gee import sentinel1, sentinel2


# dataset for urban extraction with building footprints
class UrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False):
        super().__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.root_dir = Path(cfg.DATASETS.PATH)

        if dataset == 'train':
            self.transform = compose_transformations(cfg)
        else:
            self.transform = transforms.Compose([Numpy2Torch()])

        self.include_projection = include_projection

        # loading metadata of dataset
        self.dataset = dataset
        with open(str(self.root_dir / f'{dataset}.json')) as f:
            metadata = json.load(f)
        self.metadata = metadata

        self.length = len(self.metadata['samples'])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        available_features_sentinel1 = metadata['sentinel1_features']
        selected_features_sentinel1 = cfg.DATALOADER.SENTINEL1.POLARIZATIONS
        self.s1_feature_selection = self._get_feature_selection(available_features_sentinel1,
                                                                selected_features_sentinel1)

        available_features_sentinel2 = metadata['sentinel2_features']
        selected_features_sentinel2 = cfg.DATALOADER.SENTINEL2.BANDS
        self.s2_feature_selection = self._get_feature_selection(available_features_sentinel2,
                                                                selected_features_sentinel2)

    def __getitem__(self, index):

        # loading metadata of sample
        sample_metadata = self.metadata['samples'][index]

        city = sample_metadata['city']
        patch_id = sample_metadata['patch_id']

        img, geotransform, crs = self._get_sentinel_data(city, patch_id)

        label, geotransform, crs = self._get_label_data(city, patch_id)
        img, label = self.transform((img, label))

        sample = {
            'x': img,
            'y': label,
            'city': city,
            'patch_id': patch_id,
            'image_weight': np.float(sample_metadata['img_weight'])
        }

        if self.include_projection:
            sample['transform'] = geotransform
            sample['crs'] = crs

        return sample

    def _get_sentinel_data(self, city, patch_id):

        s1_file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{patch_id}.tif'
        s2_file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{patch_id}.tif'

        # loading images and corresponding label
        if not any(self.s1_feature_selection):  # only sentinel 2 features
            img, transform, crs = read_tif(s2_file)
            img = img[:, :, self.s2_feature_selection]
        elif not any(self.s2_feature_selection):  # only sentinel 1 features
            img, transform, crs = read_tif(s1_file)
            img = img[:, :, self.s1_feature_selection]
        else:  # sentinel 1 and sentinel 2 features
            s1_img, transform, crs = read_tif(s1_file)
            s1_img = s1_img[:, :, self.s1_feature_selection]
            s2_img, transform, crs = read_tif(s2_file)
            s2_img = s2_img[:, :, self.s2_feature_selection]
            img = np.concatenate([s1_img, s2_img], axis=-1)

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, city, patch_id):

        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_dir / city / label / f'{label}_{city}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_feature_selection(self, features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def get_index(self, city: str, patch_id: str):

        samples = self.metadata['samples']

        for index, sample in enumerate(samples):
            if sample['city'] == city and sample['patch_id'] == patch_id:
                return index
        return None

    def classify_item(self, index, trained_net, device):

        # getting item
        item = self.__getitem__(index)
        img = item['x'].to(device)

        # applying trained network to item
        y_pred = trained_net(img.unsqueeze(0))
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().detach().numpy()

        # applying threshold
        y_pred = y_pred[0,] > self.cfg.THRESH
        y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')

        return y_pred, item.get('transform'), item.get('crs')

    def __len__(self):
        return self.length


# dataset for classifying a scene
class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, s1_file: Path = None, s2_file: Path = None, patch_size: int = 256,
                 s1_bands: list = None, s2_bands: list = None):
        super().__init__()

        self.cfg = cfg
        self.s1_file = s1_file
        self.s2_file = s2_file

        self.transform = transforms.Compose([Numpy2Torch()])

        ref_file = s1_file if s1_file is not None else s2_file
        arr, self.geotransform, self.crs = read_tif(ref_file)
        self.height, self.width, _ = arr.shape

        self.patch_size = patch_size
        self.rf = 8
        self.n_rows = (self.height - self.rf) // patch_size
        self.n_cols = (self.width - self.rf) // patch_size
        self.length = self.n_rows * self.n_cols

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        if s1_bands is None:
            s1_bands = ['VV', 'VH']
        selected_features_sentinel1 = cfg.DATALOADER.SENTINEL1.POLARIZATIONS
        self.s1_feature_selection = self._get_feature_selection(s1_bands, selected_features_sentinel1)

        if s2_bands is None:
            s2_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        selected_features_sentinel2 = cfg.DATALOADER.SENTINEL2.BANDS
        self.s2_feature_selection = self._get_feature_selection(s2_bands, selected_features_sentinel2)

        # loading image
        if not any(self.s1_feature_selection):  # only sentinel 2 features
            img, _, _ = read_tif(self.s2_file)
            img = img[:, :, self.s2_feature_selection]
        elif not any(self.s2_feature_selection):  # only sentinel 1 features
            img, _, _ = read_tif(self.s1_file)
            img = img[:, :, self.s1_feature_selection]
        else:  # sentinel 1 and sentinel 2 features
            s1_img, _, _ = read_tif(self.s1_file)
            s1_img = s1_img[:, :, self.s1_feature_selection]
            s2_img, _, _ = read_tif(self.s2_file)
            s2_img = s2_img[:, :, self.s2_feature_selection]
            img = np.concatenate([s1_img, s2_img], axis=-1)
        self.img = img

    def __getitem__(self, index):

        i_start = index // self.n_cols * self.patch_size
        j_start = index % self.n_cols * self.patch_size
        # check for border cases and add padding accordingly
        # top left corner
        if i_start == 0 and j_start == 0:
            i_end = self.patch_size + 2 * self.rf
            j_end = self.patch_size + 2 * self.rf
        # top
        elif i_start == 0:
            i_end = self.patch_size + 2 * self.rf
            j_end = j_start + self.patch_size + self.rf
            j_start -= self.rf
        elif j_start == 0:
            j_end = self.patch_size + 2 * self.rf
            i_end = i_start + self.patch_size + self.rf
            i_start -= self.rf
        else:
            i_end = i_start + self.patch_size + self.rf
            i_start -= self.rf
            j_end = j_start + self.patch_size + self.rf
            j_start -= self.rf

        img = self._get_sentinel_data(i_start, i_end, j_start, j_end)
        img, _ = self.transform((img, np.empty((1, 1, 1))))
        patch = {
            'x': img,
            'row': (i_start, i_end),
            'col': (j_start, j_end)
        }

        return patch

    def _get_sentinel_data(self, i_start: int, i_end: int,  j_start: int, j_end: int):
        img_patch = self.img[i_start:i_end, j_start:j_end, ]
        return np.nan_to_num(img_patch).astype(np.float32)

    def _get_feature_selection(self, features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def __len__(self):
        return self.length
