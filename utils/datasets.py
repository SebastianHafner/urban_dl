import torch
from torchvision import transforms

import json

from pathlib import Path

from utils.augmentations import *
from utils.geotiff import *


# dataset for urban extraction with building footprints
class UrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False, no_augmentations: bool = False,
                 include_unlabeled: bool = True):
        super().__init__()

        self.cfg = cfg
        self.root_dir = Path(cfg.DATASETS.PATH)

        self.dataset = dataset
        if dataset == 'training':
            self.sites = list(cfg.DATASETS.TRAINING)
        elif dataset == 'validation':
            self.sites = list(cfg.DATASETS.VALIDATION)
        else:  # used to load only 1 city passed as dataset
            self.sites = [dataset]

        # using parameter include_unlabeled to overwrite config
        include_unlabeled = cfg.DATALOADER.INCLUDE_UNLABELED and include_unlabeled
        if include_unlabeled:
            self.sites += cfg.DATASETS.UNLABELED

        self.no_augmentations = no_augmentations
        self.transform = transforms.Compose([Numpy2Torch()]) if no_augmentations else compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_dir / site / 'samples.json'
            metadata = load_json(samples_file)
            samples = metadata['samples']
            # making sure unlabeled data is not used as labeled when labels exist
            if include_unlabeled and site in cfg.DATASETS.UNLABELED:
                for sample in samples:
                    sample['is_labeled'] = False
            self.samples += samples
            s1_bands = metadata['sentinel1_features']
            s2_bands = metadata['sentinel2_features']
        self.length = len(self.samples)
        self.n_labeled = len([s for s in self.samples if s['is_labeled']])

        self.include_projection = include_projection

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]

        site = sample['site']
        patch_id = sample['patch_id']
        is_labeled = sample['is_labeled']

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, geotransform, crs = self._get_sentinel2_data(site, patch_id)
        elif mode == 'sar':
            img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
        else:  # fusion baby!!!
            s1_img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(site, patch_id)

            if self.cfg.DATALOADER.FUSION_DROPOUT and not self.no_augmentations:
                dropout_layer = np.random.randint(0, 3)
                if dropout_layer == 1:
                    s1_img[...] = 0
                if dropout_layer == 2:
                    s2_img[...] = 0

            img = np.concatenate([s1_img, s2_img], axis=-1)

        aux_inputs = self.cfg.DATALOADER.AUXILIARY_INPUTS
        for aux_input in aux_inputs:
            aux_img, _, _ = self._get_auxiliary_data(aux_input, site, patch_id)
            img = np.concatenate([aux_img, img], axis=-1)

        if is_labeled:
            label, _, _ = self._get_label_data(site, patch_id)
        else:
            label = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'site': site,
            'patch_id': patch_id,
            'is_labeled': sample['is_labeled'],
            'image_weight': np.float(sample['img_weight']),
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
        labeled_perc = self.n_labeled / self.length * 100
        return f'Dataset with {self.length} samples ({labeled_perc:.1f} % labeled) across {len(self.sites)} sites.'


# dataset for urban extraction with building footprints
class MTUrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False, no_augmentations: bool = False):
        super().__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.root_dir = Path(cfg.DATASETS.PATH)

        self.dataset = dataset
        if dataset == 'training':
            self.sites = list(cfg.DATASETS.TRAINING)
        elif dataset == 'validation':
            self.sites = list(cfg.DATASETS.VALIDATION)
        else:  # used to load only 1 city passed as dataset
            self.sites = [dataset]

        self.no_augmentations = no_augmentations
        self.transform = transforms.Compose([Numpy2Torch()]) if no_augmentations else compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_dir / site / 'samples.json'
            metadata = load_json(samples_file)
            self.samples += metadata['samples']
            s1_bands = metadata['sentinel1_features']
            s2_bands = metadata['sentinel2_features']
        self.length = len(self.samples)
        self.n_labeled = len([s for s in self.samples if s['is_labeled']])

        self.include_projection = include_projection

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]

        site = sample['site']
        patch_id = sample['patch_id']
        label_exists = sample['is_labeled']

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, geotransform, crs = self._get_sentinel2_data(site, patch_id)
        elif mode == 'sar':
            img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
        else:  # fusion baby!!!
            s1_img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(site, patch_id)

            if self.cfg.DATALOADER.FUSION_DROPOUT and not self.no_augmentations:
                dropout_layer = np.random.randint(0, 3)
                if dropout_layer == 1:
                    s1_img[...] = 0
                if dropout_layer == 2:
                    s2_img[...] = 0

            img = np.concatenate([s1_img, s2_img], axis=-1)

        aux_inputs = self.cfg.DATALOADER.AUXILIARY_INPUTS
        for aux_input in aux_inputs:
            aux_img, _, _ = self._get_auxiliary_data(aux_input, site, patch_id)
            img = np.concatenate([aux_img, img], axis=-1)

        if label_exists:
            label, _, _ = self._get_label_data(site, patch_id)
        else:
            label = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'is_labeled': label_exists,
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

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        lperc = self.n_labeled / self.length * 100
        return f'Dataset with {self.length} samples ({lperc:.1f} % labeled) across {len(self.sites)} sites.'


# dataset for urban extraction with building footprints
class STUrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, sar_cfg, run_type: str, no_augmentations: bool = False):
        super().__init__()

        self.transform = transforms.Compose([Numpy2Torch()]) if no_augmentations else compose_transformations(cfg)

        self.dataset = UrbanExtractionDataset(cfg, run_type, no_augmentations=True, include_unlabeled=True)
        sar_cfg.DATASETS = cfg.DATASETS
        self.sar_dataset = UrbanExtractionDataset(sar_cfg, run_type, no_augmentations=True, include_unlabeled=True)

        # samples need to be identical
        assert(len(self.dataset) == len(self.sar_dataset))
        self.length = len(self.dataset)
        self.sites = self.dataset.sites

    def __getitem__(self, index):

        # loading metadata of sample
        item = self.dataset.__getitem__(index)
        item_sar = self.sar_dataset.__getitem__(index)

        img = item['x'].numpy().transpose((1, 2, 0))
        split_index = img.shape[-1]
        img_sar = item_sar['x'].numpy().transpose((1, 2, 0))
        label = item['y'].numpy().transpose((1, 2, 0))

        img_concat, label = self.transform((np.concatenate((img, img_sar), axis=-1), label))
        img = img_concat[:split_index, ]
        img_sar = img_concat[split_index:, ]

        item = {
            'x': img,
            'x_sar': img_sar,
            'y': label,
            'is_labeled': item['is_labeled'],
            'site': item['site'],
            'patch_id': item['patch_id'],
        }

        return item

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
        metadata = load_json(samples_file)
        self.samples = metadata['samples']
        self.group_names = metadata['group_names']
        self.length = len(self.samples)
        s1_bands = metadata['sentinel1_features']
        s2_bands = metadata['sentinel2_features']

        self.transform = transforms.Compose([Numpy2Torch()])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
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
class SceneInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, s1_file: Path = None, s2_file: Path = None, patch_size: int = 256,
                 s1_bands: list = None, s2_bands: list = None):
        super().__init__()

        self.cfg = cfg
        self.s1_file = s1_file
        self.s2_file = s2_file
        assert(s1_file.exists() and s2_file.exists())

        self.transform = transforms.Compose([Numpy2Torch()])

        ref_file = s1_file if s1_file is not None else s2_file
        arr, self.geotransform, self.crs = read_tif(ref_file)
        self.height, self.width, _ = arr.shape

        self.patch_size = patch_size
        self.rf = 128
        self.n_rows = (self.height - self.rf) // patch_size
        self.n_cols = (self.width - self.rf) // patch_size
        self.length = self.n_rows * self.n_cols

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        if s1_bands is None:
            s1_bands = ['VV_mean', 'VV_stdDev', 'VH_mean', 'VH_stdDev']
        selected_features_sentinel1 = cfg.DATALOADER.SENTINEL1_BANDS
        self.s1_feature_selection = self._get_feature_selection(s1_bands, selected_features_sentinel1)
        if s2_bands is None:
            s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        selected_features_sentinel2 = cfg.DATALOADER.SENTINEL2_BANDS
        self.s2_feature_selection = self._get_feature_selection(s2_bands, selected_features_sentinel2)

        # loading image
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = read_tif(self.s2_file)
            img = img[:, :, self.s2_feature_selection]
        elif mode == 'sar':
            img, _, _ = read_tif(self.s1_file)
            img = img[:, :, self.s1_feature_selection]
        else:  # fusion
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

    @ staticmethod
    def _get_feature_selection(features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def get_mask(self, data_type = 'uint8') -> np.ndarray:
        mask = np.empty(shape=(self.n_rows * self.patch_size, self.n_cols * self.patch_size, 1), dtype=data_type)
        return mask

    def __len__(self):
        return self.length


# dataset for classifying a scene
class TilesInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, site: str):
        super().__init__()

        self.cfg = cfg
        self.site = site
        self.root_dir = Path(cfg.DATASETS.PATH)
        self.transform = transforms.Compose([Numpy2Torch()])

        # getting all files
        samples_file = self.root_dir / site / 'samples.json'
        metadata = load_json(samples_file)
        self.samples = metadata['samples']
        self.length = len(self.samples)

        self.patch_size = metadata['patch_size']

        # computing extent
        patch_ids = [s['patch_id'] for s in self.samples]
        self.coords = [[int(c) for c in patch_id.split('-')] for patch_id in patch_ids]
        self.max_y = max([c[0] for c in self.coords])
        self.max_x = max([c[1] for c in self.coords])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(metadata['sentinel1_features'], cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(metadata['sentinel2_features'], cfg.DATALOADER.SENTINEL2_BANDS)
        if cfg.DATALOADER.MODE == 'sar':
            self.n_features = len(self.s1_indices)
        elif cfg.DATALOADER.MODE == 'optical':
            self.n_features = len(self.s2_indices)
        else:
            self.n_features = len(self.s1_indices) + len(self.s2_indices)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        patch_id_center = sample['patch_id']

        y_center, x_center = patch_id_center.split('-')
        y_center, x_center = int(y_center), int(x_center)

        extended_patch = np.zeros((3 * self.patch_size, 3 * self.patch_size, self.n_features), dtype=np.float32)

        for i in range(3):
            for j in range(3):
                y = y_center + (i - 1) * self.patch_size
                x = x_center + (j - 1) * self.patch_size
                patch_id = f'{y:010d}-{x:010d}'
                if self._is_valid_patch_id(patch_id):
                    patch = self._load_patch(patch_id)
                else:
                    patch = np.zeros((self.patch_size, self.patch_size, self.n_features), dtype=np.float32)
                i_start = i * self.patch_size
                i_end = (i + 1) * self.patch_size
                j_start = j * self.patch_size
                j_end = (j + 1) * self.patch_size
                extended_patch[i_start:i_end, j_start:j_end, :] = patch

        dummy_label = np.zeros((extended_patch.shape[0], extended_patch.shape[1], 1), dtype=np.float32)
        extended_patch, _ = self.transform((extended_patch, dummy_label))

        item = {
            'x': extended_patch,
            'i': y_center,
            'j': x_center,
            'site': self.site,
            'patch_id': patch_id_center,
        }

        return item

    def _is_valid_patch_id(self, patch_id):
        patch_ids = [s['patch_id'] for s in self.samples]
        return True if patch_id in patch_ids else False

    def _load_patch(self, patch_id):
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(patch_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(patch_id)
        else:  # fusion baby!!!
            s1_img, _, _ = self._get_sentinel1_data(patch_id)
            s2_img, _, _ = self._get_sentinel2_data(patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)
        return img

    def _get_sentinel1_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel1' / f'sentinel1_{self.site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel2' / f'sentinel2_{self.site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.patch_size
        width = self.max_x + self.patch_size
        return np.zeros((height, width, 1), dtype=dtype)

    def get_geo(self):
        patch_id = f'{0:010d}-{0:010d}'
        _, transform, crs = self._get_sentinel1_data(patch_id)
        return transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'

