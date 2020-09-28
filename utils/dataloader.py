import torch
from torchvision import transforms

import json

from pathlib import Path

from utils.augmentations import *
from utils.geotiff import *


# dataset for urban extraction with building footprints
class UrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False, no_augmentations: bool = False):
        super().__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.root_dir = Path(cfg.DATASETS.PATH)

        self.dataset = dataset
        if dataset == 'training':
            self.sites = cfg.DATASETS.SITES.TRAINING
        elif dataset == 'validation':
            self.sites = cfg.DATASETS.SITES.VALIDATION
        else:  # used to load only 1 city passed as dataset
            self.sites = [dataset]

        self.no_augmentations = no_augmentations
        self.transform = transforms.Compose([Numpy2Torch()]) if no_augmentations else compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_dir / site / 'samples.json'
            with open(str(samples_file)) as f:
                metadata = json.load(f)
            self.samples += metadata['samples']
        self.length = len(self.samples)

        self.include_projection = include_projection

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = ['VV_mean', 'VV_stdDev', 'VH_mean', 'VH_stdDev']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]

        site = sample['site']
        patch_id = sample['patch_id']

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(site, patch_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(site, patch_id)
        else:  # fusion baby!!!
            s1_img, _, _ = self._get_sentinel1_data(site, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(site, patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)

        aux_inputs = self.cfg.DATALOADER.AUXILIARY_INPUTS
        for aux_input in aux_inputs:
            aux_img, _, _ = self._get_auxiliary_data(aux_input, site, patch_id)
            img = np.concatenate([aux_img, img], axis=-1)

        label, geotransform, crs = self._get_label_data(site, patch_id)
        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'site': site,
            'patch_id': patch_id,
            'image_weight': np.float(sample['img_weight'])
        }

        if self.include_projection:
            item['transform'] = geotransform
            item['crs'] = crs

        return item

    def _get_sentinel1_data(self, site, patch_id):
        file = self.root_dir / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, site, patch_id):
        file = self.root_dir / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_auxiliary_data(self, aux_input, site, patch_id):
        file = self.root_dir / site / aux_input / f'{aux_input}_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, site, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_dir / site / label / f'{label}_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    @ staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'


# dataset for urban extraction with building footprints
class SpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.root_dir = Path(cfg.DATASETS.TESTING)

        # getting patches
        samples_file = self.root_dir / 'samples.json'
        with open(str(samples_file)) as f:
            metadata = json.load(f)
        self.samples = metadata['samples']
        self.group_names = metadata['group_names']
        self.length = len(self.samples)

        self.transform = transforms.Compose([Numpy2Torch()])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = ['VV', 'VH']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        aoi_id = sample['aoi_id']
        group = sample['group']
        group_name = self.group_names[str(group)]

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(aoi_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(aoi_id)
        else:  # fusion baby!!!
            s1_img, _, _ = self._get_sentinel1_data(aoi_id)
            s2_img, _, _ = self._get_sentinel2_data(aoi_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)

        aux_inputs = self.cfg.DATALOADER.AUXILIARY_INPUTS
        for aux_input in aux_inputs:
            aux_img, _, _ = self._get_auxiliary_data(aux_input, aoi_id)
            img = np.concatenate([aux_img, img], axis=-1)

        label, geotransform, crs = self._get_label_data(aoi_id)
        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'aoi_id': aoi_id,
            'country': sample['country'],
            'group': group,
            'group_name': group_name,
            'year': sample['year'],
            'month': sample['month'],
            'transform': geotransform,
            'crs': crs
        }

        return item

    def _get_sentinel1_data(self, aoi_id):
        file = self.root_dir / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, aoi_id):
        file = self.root_dir / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, aoi_id):

        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_dir / label / f'{label}_{aoi_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_auxiliary_data(self, aux_input, aoi_id):
        file = self.root_dir / aux_input / f'{aux_input}_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'


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
        selected_features_sentinel1 = cfg.DATALOADER.SENTINEL1_BANDS
        self.s1_feature_selection = self._get_feature_selection(s1_bands, selected_features_sentinel1)

        if s2_bands is None:
            s2_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        selected_features_sentinel2 = cfg.DATALOADER.SENTINEL2_BANDS
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

    def _get_sentinel_data(self, i_start: int, i_end: int, j_start: int, j_end: int):
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
