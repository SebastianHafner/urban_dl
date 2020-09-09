import shutil, json
from pathlib import Path
from utils.geotiff import *
import numpy as np
from tqdm import tqdm


# computing the percentage of urban pixels for a file
def get_image_weight(file: Path):
    if not file.exists():
        raise FileNotFoundError(f'Cannot find file {file.name}')
    arr, _, _ = read_tif(file)
    n_urban = np.sum(arr)
    return int(n_urban)


def has_only_zeros(file: Path) -> bool:
    arr, _, _ = read_tif(file)
    sum_ = np.sum(arr)
    if sum_ == 0:
        return True
    return False


def crop_patch(file: Path, patch_size: int):
    arr, transform, crs = read_tif(file)
    i, j, _ = arr.shape
    if i > patch_size or j > patch_size:
        arr = arr[:patch_size, :patch_size, ]
        write_tif(file, arr, transform, crs)
    elif i < patch_size or j < patch_size:
        raise Exception(f'invalid file found {file.name}')
    else:
        pass


def preprocess(path: Path, site: str, patch_size: int = 256):

    print(f'preprocessing {site}')

    sentinel1_dir = path / 'sentinel1'
    sentinel2_dir = path / 'sentinel2'
    buildings_dir = path / 'buildings'
    buildings_files = [file for file in buildings_dir.glob('**/*')]

    samples = []
    for i, buildings_file in enumerate(tqdm(buildings_files)):

        _, _, patch_id = buildings_file.stem.split('_')
        crop_patch(buildings_file, patch_size)

        sentinel1_file = sentinel1_dir / f'sentinel1_{site}_{patch_id}.tif'
        crop_patch(sentinel1_file, patch_size)
        if has_only_zeros(sentinel1_file):
            raise Exception(f'only zeros {sentinel1_file.name}')

        sentinel2_file = sentinel2_dir / f'sentinel2_{site}_{patch_id}.tif'
        crop_patch(sentinel2_file, patch_size)
        if has_only_zeros(sentinel2_file):
            raise Exception(f'only zeros {sentinel2_file.name}')

        sample = {
            'site': site,
            'patch_id': patch_id,
            'img_weight': get_image_weight(buildings_file)
        }
        samples.append(sample)

    # writing data to json file
    data = {
        'label': 'buildings',
        'site': site,
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





