import shutil, json
from pathlib import Path
from utils.geotiff import *
import numpy as np


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


# preprocessing dataset
def preprocess_dataset(data_dir: Path, save_dir: Path, cities: list, year: int, labels: str,
                       s1_features: list, s2_features: list, split: float, seed: int = 42):

    # setting up save dir
    if not save_dir.exists():
        save_dir.mkdir()

    # container to store all the metadata
    dataset_metadata = {
        'cities': cities,
        'year': year,
        'sentinel1': s1_features,
        'sentinel2': s2_features,
        'labels': labels
    }

    # getting label files from first label (used edge detection in loop)
    label_dir = data_dir / labels[0]
    label_files = [file for file in label_dir.glob('**/*')]

    # generating random numbers for split
    np.random.seed(seed)
    random_numbers = list(np.random.uniform(size=len(label_files)))

    # main loop splitting into train test, removing edge tiles, and collecting metadata
    samples = {'train': [], 'test': []}
    for i, (label_file, random_num) in enumerate(zip(label_files, random_numbers)):
        if not is_edge_tile(label_file):

            sample_metadata = {}

            _, city, patch_id = label_file.stem.split('_')

            sample_metadata['city'] = city
            sample_metadata['patch_id'] = patch_id
            sample_metadata['img_weight'] = get_image_weight(label_file)

            if random_num > split:
                train_test_dir = save_dir / 'train'
                samples['train'].append(sample_metadata)
            else:
                train_test_dir = save_dir / 'test'
                samples['test'].append(sample_metadata)

            if not train_test_dir.exists():
                train_test_dir.mkdir()

            # copying all files into new directory
            for j, product in enumerate(['sentinel1', 'sentinel2', *labels]):

                if product == 'sentinel1':
                    file_name = f'S1_{city}_{year}_{patch_id}.tif'
                elif product == 'sentinel2':
                    file_name = f'S2_{city}_{year}_{patch_id}.tif'
                else:  # for all labels
                    file_name = f'{product}_{city}_{patch_id}.tif'

                src_file = data_dir / product / file_name
                new_file = train_test_dir / product / file_name
                if not new_file.parent.exists():
                    new_file.parent.mkdir()
                shutil.copy(str(src_file), str(new_file))


    # writing metadata to .json file for train and test set
    for train_test in ['train', 'test']:
        dataset_metadata['dataset'] = train_test
        dataset_metadata['samples'] = samples[train_test]
        metadata_file = save_dir / train_test / 'metadata.json'
        with open(str(metadata_file), 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)


def create_train_test_split(root_dir: Path, split: float = 0.3, delete_edge_files: bool = False, seed: int = None):

    # loading metadata from download
    dl_file = root_dir / 'download_metadata.json'
    assert(dl_file.exists())
    with open(str(dl_file)) as f:
        download = json.load(f)

    # unpacking required data
    year = download['year']
    labels = download['labels']
    polarizations = download['sentinel1']['polarizations']
    orbits = download['sentinel1']['orbits']
    s1_metrics = download['sentinel1']['metrics']
    bands = download['sentinel2']['bands']
    indices = download['sentinel2']['indices']
    s2_metrics = download['sentinel2']['metrics']

    # getting label files from first label (used edge detection in loop)
    label_dir = root_dir / labels[0]
    label_files = [file for file in label_dir.glob('**/*')]

    # generating random numbers for split
    np.random.seed(seed)
    random_numbers = list(np.random.uniform(size=len(label_files)))

    # main loop splitting into train test, removing edge tiles, and collecting metadata
    samples = {'train': [], 'test': [], 'inference': []}
    for i, (label_file, random_num) in enumerate(zip(label_files, random_numbers)):

        _, city, patch_id = label_file.stem.split('_')

        if not is_edge_tile(label_file):

            sample_metadata = {
                'city': city,
                'patch_id': patch_id,
                'img_weight': get_image_weight(label_file)
            }

            if random_num > split:
                samples['train'].append(sample_metadata)
                sample_metadata['dataset'] = 'train'
            else:
                samples['test'].append(sample_metadata)
                sample_metadata['dataset'] = 'test'

            samples['inference'].append(sample_metadata)

        else:
            if delete_edge_files:
                # deleting all edge files
                for j, product in enumerate(['sentinel1', 'sentinel2', *labels]):

                    if product == 'sentinel1':
                        file_name = f'{product}_{city}_{year}_{patch_id}.tif'
                    elif product == 'sentinel2':
                        file_name = f'{product}_{city}_{year}_{patch_id}.tif'
                    else:  # for all labels
                        file_name = f'{product}_{city}_{patch_id}.tif'

                    edge_file = root_dir / product / file_name
                    edge_file.unlink()

    # writing metadata to .json file for train and test set
    s1_features = sentinel1.get_feature_names(polarizations, orbits, s1_metrics)
    s2_features = sentinel2.get_feature_names(bands, indices, s2_metrics)
    for dataset in ['train', 'test', 'inference']:
        data = {
            'labels': download['labels'],
            'cities': download['cities'],
            'year': download['year'],
            'sentinel1_features': s1_features,
            'sentinel2_features': s2_features,
            'dataset': dataset,
            'samples': samples[dataset]
        }
        dataset_file = root_dir / f'{dataset}.json'
        with open(str(dataset_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def create_inference_file(root_dir: Path, delete_edge_files: bool = False):

    # loading metadata from download
    dl_file = root_dir / 'download_metadata.json'
    assert(dl_file.exists())
    with open(str(dl_file)) as f:
        download = json.load(f)

    # unpacking required data
    year = download['year']
    polarizations = download['sentinel1']['polarizations']
    orbits = download['sentinel1']['orbits']
    s1_metrics = download['sentinel1']['metrics']
    bands = download['sentinel2']['bands']
    indices = download['sentinel2']['indices']
    s2_metrics = download['sentinel2']['metrics']

    # getting label files from first label (used edge detection in loop)
    s1_dir = root_dir / 'sentinel1'
    s1_files = [file for file in s1_dir.glob('**/*')]

    # main loop removing edge tiles, and collecting metadata
    samples = []
    for i, s1_file in enumerate(s1_files):

        _, city, _, patch_id = s1_file.stem.split('_')

        if not is_edge_tile(s1_file):
            print(f'Non-edge tile: {city}_{patch_id}')

            sample_metadata = {
                'city': city,
                'patch_id': patch_id,
            }
            samples.append(sample_metadata)

        else:
            print(f'Edge tile: {city}_{patch_id}')
            if delete_edge_files:
                # deleting all edge files
                for product in ['sentinel1', 'sentinel2']:

                    file_name = f'{product}_{city}_{year}_{patch_id}.tif'
                    edge_file = root_dir / product / file_name
                    edge_file.unlink()

    # writing metadata to .json file for train and test set
    s1_features = sentinel1.get_feature_names(polarizations, orbits, s1_metrics)
    s2_features = sentinel2.get_feature_names(bands, indices, s2_metrics)
    for dataset in ['train', 'test', 'inference']:
        data = {
            'labels': download['labels'],
            'cities': download['cities'],
            'year': download['year'],
            'sentinel1_features': s1_features,
            'sentinel2_features': s2_features,
            'dataset': dataset,
            'samples': samples
        }
        inference_file = root_dir / 'inference.json'
        with open(str(inference_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def write_metadata_file(root_dir: Path, year: int, cities: list, s1_features: list, s2_features: list):

    # TODO: use this function also in the main preprocessing function to generate metadata file

    # setting up raw data directories
    s1_dir = root_dir / 'sentinel1'

    # container to store all the metadata
    dataset_metadata = {
        'cities': cities,
        'year': year,
        'sentinel1': s1_features,
        'sentinel2': s2_features,
    }

    # getting all sentinel1 files
    s1_files = [file for file in s1_dir.glob('**/*')]

    # main loop splitting into train test, removing edge tiles, and collecting metadata
    samples = []
    for i, s1_file in enumerate(s1_files):
        if not is_edge_tile(s1_file):

            sample_metadata = {}

            _, city, _, patch_id = s1_file.stem.split('_')

            sample_metadata['city'] = city
            sample_metadata['patch_id'] = patch_id

            samples.append(sample_metadata)

    # writing metadata to .json file for train and test set
    dataset_metadata['samples'] = samples
    with open(str(root_dir / 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)


def process_city(root_dir: Path, city: str, patch_size: int = 256):

    sentinel1_dir = root_dir / city / 'sentinel1'
    sentinel2_dir = root_dir / city / 'sentinel2'
    buildings_dir = root_dir / city / 'buildings'
    buildings_files = [file for file in buildings_dir.glob('**/*')]

    samples = []
    for i, buildings_file in enumerate(buildings_files):
        _, _, patch_id = buildings_file.stem.split('_')
        print(f'{city} {patch_id}')

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
            'sentinel2_features': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            'dataset': dataset,
            'samples': samples
        }
        dataset_file = root_dir / f'{dataset}.json'
        with open(str(dataset_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def preprocess(path: Path, city: str, patch_size: int = 256):

    sentinel1_dir = path / 'sentinel1'
    sentinel2_dir = path / 'sentinel2'
    buildings_dir = path / 'buildings'
    buildings_files = [file for file in buildings_dir.glob('**/*')]

    samples = []
    for i, buildings_file in enumerate(buildings_files):
        _, _, patch_id = buildings_file.stem.split('_')
        print(f'{city} {patch_id}')

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
        'sentinel2_features': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        'samples': samples
    }
    dataset_file = path / f'samples.json'
    with open(str(dataset_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)





if __name__ == '__main__':

    root_dir = Path('/storage/shafner/urban_extraction/urban_extraction_buildings/')
    dataset_path = Path('C:/Users/shafner/urban_extraction/data/dummy_data')

    train_cities = ['dallas', 'miami', 'vancouver', 'toronto', 'newyork', 'dallas', 'kampala']
    test_cities = ['losangeles', 'daressalaam']

    cities = ['miami', 'houston']
    for city in cities:
        path = dataset_path / city
        preprocess(path, city, 256)

    # create_city_split(root_dir, train_cities=train_cities, test_cities=test_cities)
    # create_train_test_split(root_dir, split=0.1, seed=7, delete_edge_files=True)
    # create_inference_file(root_dir, delete_edge_files=True)

    # cities = ['NewYork']
    # year = 2017
    # labels = ['bing', 'wsf']
    # bucket = 'urban_extraction_bing_raw'
    # data_dir = gee_dir / bucket
    # save_dir = save_dir / bucket[:-4]
    #
    #
    # split = 0.2
    #
    # # sentinel 1 parameters
    # s1params = {
    #     'polarizations': ['VV', 'VH'],
    #     'metrics': ['mean']
    # }
    #
    # # sentinel 2 parameters
    # s2params = {
    #     'bands': ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'],
    #     'indices': [],
    #     'metrics': ['median']
    # }
    #
    # # generating feature names for sentinel 1 and sentinel 2
    # sentinel1_features = sentinel1_feature_names(polarizations=s1params['polarizations'],
    #                                              metrics=s1params['metrics'])
    # sentinel2_features = sentinel2_feature_names(bands=s2params['bands'],
    #                                              indices=s2params['indices'],
    #                                              metrics=s2params['metrics'])

    # preprocess_dataset(data_dir, save_dir, cities, year, labels, sentinel1_features, sentinel2_features, split)

    # cities = ['Stockholm', 'Beijing', 'Milan']
    # year = 2019
    # root_dir = Path('/storage/shafner/urban_extraction/urban_extraction_2019')
    # write_metadata_file(
    #     root_dir=root_dir,
    #     year=year,
    #     cities=cities,
    #     s1_features=sentinel1_features,
    #     s2_features=sentinel2_features
    # )



