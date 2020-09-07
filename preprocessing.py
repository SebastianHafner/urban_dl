import shutil, json
from pathlib import Path
from utils.geotiff import *
import numpy as np
from tqdm import tqdm


# getting list of feature names based on input parameters
def sentinel1_feature_names(polarizations: list, metrics: list):
    names = []
    for orbit in ['asc', 'desc']:
        for pol in polarizations:
            for metric in metrics:
                names.append(f'{pol}_{orbit}_{metric}')
    return names


# getting list of feature names based on input parameters
def sentinel2_feature_names(bands: list, indices: list, metrics: list):
    band_names = [f'{band}_{metric}' for band in bands for metric in metrics]
    index_names = [f'{index}_{metric}' for index in indices for metric in metrics]
    return band_names + index_names


# computing the percentage of urban pixels for a file
def get_image_weight(file: Path):
    if not file.exists():
        raise FileNotFoundError(f'Cannot find file {file.name}')
    arr, _, _ = read_tif(file)
    n_urban = np.sum(arr)
    return int(n_urban)


def is_edge_tile(file: Path, tile_size=256):
    arr, _, _ = read_tif(file)
    arr = np.array(arr)
    if arr.shape[0] == tile_size and arr.shape[1] == tile_size:
        return False
    return True


def process_city(root_dir: Path, city: str, patch_size: int = 256):

    print(f'processing {city}')

    sentinel1_dir = root_dir / city / 'sentinel1'
    sentinel2_dir = root_dir / city / 'sentinel2'
    buildings_dir = root_dir / city / 'buildings'
    buildings_files = [file for file in buildings_dir.glob('**/*')]

    samples = []
    for i, buildings_file in enumerate(tqdm(buildings_files)):
        _, _, patch_id = buildings_file.stem.split('_')

        sentinel1_file = sentinel1_dir / f'sentinel1_{city}_{patch_id}.tif'
        sentinel2_file = sentinel2_dir / f'sentinel2_{city}_{patch_id}.tif'

        for file in [buildings_file, sentinel1_file, sentinel2_file]:
            arr, transform, crs = read_tif(file)
            i, j, _ = arr.shape
            if i > patch_size or j > patch_size:
                arr = arr[:patch_size, :patch_size, ]
                write_tif(file, arr, transform, crs)
            elif i < patch_size or j < patch_size:
                raise Exception(f'invalid file found {buildings_file.name}')
            else:
                pass

        sample = {
            'city': city,
            'patch_id': patch_id,
            'img_weight': get_image_weight(buildings_file)
        }
        samples.append(sample)

    return samples


def create_city_split(root_dir: Path, train_cities: list, test_cities: list):

    for dataset in ['train', 'test']:
        cities = train_cities if dataset == 'train' else test_cities

        samples = []
        for city in cities:
            city_samples = process_city(root_dir, city, 256)
            samples = samples + city_samples

        # writing metadata to .json file for train and test set
        data = {
            'label': 'buildings',
            'cities': cities,
            'sentinel1_features': ['VV', 'VH'],
            'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
            'dataset': dataset,
            'samples': samples
        }
        dataset_file = root_dir / f'{dataset}.json'
        with open(str(dataset_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# TODO: add progress bar and clean up functions (tis one seems to be similar to process city)
def preprocess(path: Path, city: str, patch_size: int = 256):

    print(f'preprocessing {city}')

    sentinel1_dir = path / 'sentinel1'
    sentinel2_dir = path / 'sentinel2'
    buildings_dir = path / 'buildings'
    buildings_files = [file for file in buildings_dir.glob('**/*')]

    samples = []
    for i, buildings_file in enumerate(tqdm(buildings_files)):
        _, _, patch_id = buildings_file.stem.split('_')

        sentinel1_file = sentinel1_dir / f'sentinel1_{city}_{patch_id}.tif'
        sentinel2_file = sentinel2_dir / f'sentinel2_{city}_{patch_id}.tif'

        for file in [buildings_file, sentinel1_file, sentinel2_file]:
            arr, transform, crs = read_tif(file)
            i, j, _ = arr.shape
            if i > patch_size or j > patch_size:
                arr = arr[:patch_size, :patch_size, ]
                write_tif(file, arr, transform, crs)
            elif i < patch_size or j < patch_size:
                raise Exception(f'invalid file found {buildings_file.name}')
            else:
                pass

        sample = {
            'city': city,
            'patch_id': patch_id,
            'img_weight': get_image_weight(buildings_file)
        }
        samples.append(sample)

    # writing data to json file
    data = {
        'label': 'buildings',
        'city': city,
        'sentinel1_features': ['VV', 'VH'],
        'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'samples': samples
    }
    dataset_file = path / f'samples.json'
    with open(str(dataset_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def sites_split(sites: list, train_fraction: float):
    n = len(sites)
    n_train = int(n * train_fraction)
    print(n_train)
    training_sites = list(np.random.choice(sites, size=n_train, replace=False))
    validation_sites = [site for site in sites if site not in training_sites]
    print(training_sites, validation_sites)


if __name__ == '__main__':

    # dataset_path = Path('C:/Users/shafner/urban_extraction/data/dummy_data')
    dataset_path = Path('/storage/shafner/urban_extraction/urban_extraction/')

    training = ['denver', 'saltlakecity', 'phoenix', 'lasvegas', 'toronto', 'columbus', 'winnipeg', 'dallas',
                'minneapolis', 'atlanta', 'miami', 'montreal', 'quebec', 'albuquerque', 'losangeles', 'kansascity',
                'charlston', 'seattle', 'daressalam']
    validation = ['houston', 'sanfrancisco', 'vancouver', 'newyork', 'calgary', 'kampala']
    all_sites = training + validation
    for site in all_sites:
        path = dataset_path / site
        preprocess(path, site, 256)

    # sites_split(northamerican_sites, 0.8)

    create_city_split(dataset_path, train_cities=training, test_cities=validation)



