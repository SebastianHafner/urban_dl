import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from pathlib import Path
from utils.augmentations import *
from utils.geotiff import *



# see https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):

    def __init__(self, cfg, dataset: str, include_projection: bool = False):
        # Initialization

        self.cfg = cfg
        self.dataset = dataset
        self.root_dir = Path(cfg.DATASETS.PATH)
        self.batch_size = cfg.TRAINER.BATCH_SIZE
        self.shuffle = cfg.DATALOADER.SHUFFLE
        self.dim = (256, 256)

        self.dataset = dataset
        if dataset == 'train':
            self.cities = cfg.DATASETS.TRAIN
        elif dataset == 'test':
            self.cities = cfg.DATASETS.TEST
        else:  # used to load only 1 city passed as dataset
            self.cities = [dataset]

        self.samples = []
        for city in self.cities:
            samples_file = self.root_dir / city / 'samples.json'
            with open(str(samples_file)) as f:
                metadata = json.load(f)
            self.samples += metadata['samples']
        self.length = len(self.samples)

        self.include_projection = include_projection

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = ['VV', 'VH']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        s2_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

        self.n_channels = len(self.s1_indices) + len(self.s2_indices)

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        batch_samples = [self.samples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_samples)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_samples):
        # Generates data containing batch_size samples  X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1), dtype=int)

        # Generate data
        for i, sample in enumerate(batch_samples):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]

        city = sample['city']
        patch_id = sample['patch_id']

        # loading images
        if not any(self.cfg.DATALOADER.SENTINEL1_BANDS):  # only sentinel 2 features
            img, _, _ = self._get_sentinel2_data(city, patch_id)
        elif not any(self.cfg.DATALOADER.SENTINEL2_BANDS):  # only sentinel 1 features
            img, _, _ = self._get_sentinel1_data(city, patch_id)
        else:  # sentinel 1 and sentinel 2 features
            s1_img, _, _ = self._get_sentinel1_data(city, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(city, patch_id)

            if self.cfg.AUGMENTATION.SENSOR_DROPOUT and self.dataset == 'train':
                if np.random.rand() > self.cfg.AUGMENTATION.DROPOUT_PROBABILITY:
                    no_optical = np.random.choice([True, False])
                    if no_optical:
                        s2_img = np.zeros(s2_img.shape)
                    else:
                        s1_img = np.zeros(s1_img.shape)

            img = np.concatenate([s1_img, s2_img], axis=-1)

        label, geotransform, crs = self._get_label_data(city, patch_id)
        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'city': city,
            'patch_id': patch_id,
            'image_weight': np.float(sample['img_weight'])
        }

        if self.include_projection:
            item['transform'] = geotransform
            item['crs'] = crs

        return item

    def _get_sentinel1_data(self, city, patch_id):
        file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, city, patch_id):
        file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, city, patch_id):

        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_dir / city / label / f'{label}_{city}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    @ staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} cities.'


