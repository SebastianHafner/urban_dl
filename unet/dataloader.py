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


class UrbanExtractionDataset(torch.utils.data.Dataset):
    '''
    Dataset for Urban Extraction style labelled Dataset
    '''
    def __init__(self, cfg, root_dir: Path, dataset: str, transform: list = None,
                 include_index: bool = False, include_projection: bool = False):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.cfg = cfg

        # loading metadata of dataset
        self.dataset = dataset
        with open(str(self.root_dir / f'{dataset}.json')) as f:
            metadata = json.load(f)
        self.metadata = metadata

        self.length = len(self.metadata['samples'])
        print('dataset length', self.length)

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([Npy2Torch()])
        self.include_index = include_index
        self.include_projection = include_projection


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
        img, label, sample_id, = self.transform((img, label, patch_id,))
        sample = {
            'x': img,
            'y': label,
            'img_name': sample_id,
            'image_weight': np.float(sample_metadata['img_weight'])
        }

        if self.include_index:
            sample['index'] = index

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

