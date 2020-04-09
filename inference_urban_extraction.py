from pathlib import Path
import rasterio
from rasterio.merge import merge

from unet import UNet
from unet import dataloader
from experiment_manager.config import new_config
from preprocessing.utils import *
from preprocessing.preprocessing_urban_extraction import write_metadata_file
from unet.augmentations import *
from torchvision import transforms

import matplotlib as mpl


# loading cfg for inference
def load_cfg(cfg_file: Path):
    cfg = new_config()
    cfg.merge_from_file(str(cfg_file))
    return cfg


# loading network for inference
def load_net(cfg, net_file):
    net = UNet(cfg)
    state_dict = torch.load(str(net_file), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    net.to(device)
    net.eval()

    return net


# loading dataset for inference
def load_dataset(cfg, root_dir: Path):
    dataset = dataloader.UrbanExtractionDataset(cfg, root_dir, 'inference', include_projection=True)
    return dataset


# getting all files of product in train and test directory
def get_train_test_files(root_dir: Path, product: str):

    train_files_dir = root_dir / 'train' / product
    test_files_dir = root_dir / 'test' / product
    files_in_train = [file for file in train_files_dir.glob('**/*')]
    files_in_test = [file for file in test_files_dir.glob('**/*')]
    files = files_in_train + files_in_test

    return files


# returns the dataset (train/test) that the patch file belongs to
def get_train_test(root_dir: Path, product, fname):

    # check for file in train
    train_file = root_dir / 'train' / product / fname
    if train_file.exists():
        return 'train'

    # check for file in test
    test_file = root_dir / 'test' / product / fname
    if test_file.exists():
        return 'test'

    return None


# uses trained model to make a prediction for each tiles
def inference_tiles(data_dir: Path, experiment: str, dataset: str, city: str, configs_dir: Path, models_dir: Path,
                    model_cp: int, metadata_exists: bool = True):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(configs_dir, f'{experiment}_{dataset}')
    net = load_net(cfg, models_dir / f'{experiment}_{dataset}', f'cp_{model_cp}')

    # setting up save directory
    save_dir = data_dir / f'pred_{experiment}_{dataset}'
    if not save_dir.exists():
        save_dir.mkdir()

    # loading dataset from config (requires metadata)
    metadata_file = data_dir / 'metadata.json'
    assert (metadata_file.exists())
    dataset = load_dataset(cfg, data_dir)
    year = dataset.metadata['year']

    # classifying all tiles in dataset
    for i in range(len(dataset)):
        item = dataset.__getitem__(i)
        img = item['x'].to(device)

        metadata = dataset.metadata['samples'][i]
        patch_city = metadata['city']
        patch_id = metadata['patch_id']
        print(patch_city, patch_id)
        if patch_city == city:
            # print(patch_city, patch_id)

            y_pred = net(img.unsqueeze(0))
            y_pred = torch.sigmoid(y_pred)

            y_pred = y_pred.cpu().detach().numpy()
            threshold = cfg.THRESH
            y_pred = y_pred[0, ] > threshold
            y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')

            file = save_dir / f'pred_{patch_city}_{year}_{patch_id}.tif'
            transform = item['transform']
            crs = item['crs']
            write_tif(file, y_pred, transform, crs)


def merge_tiles_selfmade(root_dir: Path, city: str, year: int, experiment: str, dataset: str, tile_size: int = 256,
                         show_train_test: bool = False, save_dir: Path = None):

    product = f'pred_{experiment}_{dataset}'

    # getting all files of product in train and test directory
    files = get_train_test_files(root_dir, f'pred_{experiment}_{dataset}')

    # getting width and height of mosaic by scanning all files
    height, width = 0, 0
    for file in files:
        patch_product, patch_city, patch_year, patch_id = file.stem.split('_')
        if patch_city == city:
            m, n = patch_id.split('-')
            m, n = int(m), int(n)
            if m > height:
                height = m
            if n > width:
                width = n

    height += tile_size
    width += tile_size

    mosaic = np.empty((height, width, 1), dtype=np.uint8)

    # populating arr with classification results
    for i, m in enumerate(range(0, height, tile_size)):
        for j, n in enumerate(range(0, width, tile_size)):

            patch_id = f'{m:010d}-{n:010d}'
            patch_fname = f'pred_{city}_{year}_{patch_id}.tif'

            train_test = get_train_test(root_dir, product, patch_fname)
            if train_test is None:
                print(patch_fname)
            assert(train_test is not None)

            patch_file = root_dir / train_test / product / patch_fname
            patch_data, _, _ = read_tif(patch_file)
            if show_train_test:
                if train_test == 'test':
                    patch_data *= 2
            mosaic[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, 0] = patch_data[:, :, 0]

    if save_dir is None:
        save_dir = root_dir / 'maps'
    if not save_dir.exists():
        save_dir.mkdir()

    # get transform and crs from top left patch
    m, n = 0, 0
    patch_id = f'{m:010d}-{n:010d}'
    patch_fname = f'pred_{city}_{year}_{patch_id}.tif'
    train_test = get_train_test(root_dir, product, patch_fname)
    patch_file = root_dir / train_test / product / patch_fname

    _, transform, crs = read_tif(patch_file)

    mosaic_file = save_dir / f'pred_{city}_{year}_{experiment}_{dataset}.tif'
    write_tif(mosaic_file, mosaic, transform, crs)


def merge_tiles(root_dir: Path, city: str, experiment: str, dataset: str, save_dir: Path = None):

    # getting all files of product in train and test directory
    files = get_train_test_files(root_dir, f'pred_{experiment}_{dataset}')

    # sub setting files to city
    files_to_mosaic = []
    for file in files:
        city_file = file.stem.split('_')[1]
        if city_file == city:
            src = rasterio.open(str(file))
            files_to_mosaic.append(src)

    # merging all files to a mosaic
    mosaic, out_trans = merge(files_to_mosaic)

    # getting metadata
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans})

    # dropping patch id from file name
    fname_parts = file.stem.split('_')[:-1]
    prefix = '_'.join(fname_parts)

    if save_dir is None:
        save_dir = root_dir / 'maps'
    if not save_dir.exists():
        save_dir.mkdir()

    out_file = save_dir / f'{prefix}_{experiment}.tif'
    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)


def end_to_end_inference_old(data_dir: Path, experiment: str, dataset: str, city: str, configs_dir: Path,
                         models_dir: Path, model_cp: int, tile_size: int = 256, save_dir: Path = None):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(configs_dir, f'{experiment}_{dataset}')
    net = load_net(cfg, models_dir / f'{experiment}_{dataset}', f'cp_{model_cp}')

    # loading dataset from config (requires metadata)
    metadata_file = data_dir / 'metadata.json'
    if not metadata_file.exists():
        # TODO: implement automatic metadata generation, required download file
        pass
    assert(metadata_file.exists())

    ds = load_dataset(cfg, data_dir)
    year = ds.metadata['year']

    # getting width and height of mosaic by scanning all files
    files = [file for file in (data_dir / 'sentinel1').glob('**/*')]
    height, width = 0, 0
    for file in files:
        patch_product, patch_city, patch_year, patch_id = file.stem.split('_')
        if patch_city == city:
            m, n = patch_id.split('-')
            m, n = int(m), int(n)
            if m > height:
                height = m
            if n > width:
                width = n

    height += tile_size
    width += tile_size

    mosaic = np.empty((height, width, 1), dtype=np.uint8)

    # classifying all tiles in dataset
    for i in range(len(ds)):

        sample_metadata = ds.metadata['samples'][i]
        patch_city = sample_metadata['city']
        print(patch_city)

        if patch_city == city:
            item = ds.__getitem__(i)
            img = item['x'].to(device)

            y_pred = net(img.unsqueeze(0))
            y_pred = torch.sigmoid(y_pred)

            y_pred = y_pred.cpu().detach().numpy()
            threshold = cfg.THRESH
            y_pred = y_pred[0, ] > threshold
            y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')

            patch_id = sample_metadata['patch_id']
            print(patch_id)
            m, n = patch_id.split('-')
            m, n = int(m), int(n)

            mosaic[m:m + tile_size, n:n + tile_size, 0] = y_pred[:, :, 0]

    # transform and crs from top left
    if save_dir is None:
        save_dir = data_dir / 'maps'
    if not save_dir.exists():
        save_dir.mkdir()

    # get transform and crs from top left patch
    m, n = 0, 0
    patch_id = f'{m:010d}-{n:010d}'
    patch_file = data_dir / 'sentinel1' / f'S1_{city}_{year}_{patch_id}.tif'
    _, transform, crs = read_tif(patch_file)

    mosaic_file = save_dir / f'pred_{city}_{year}_{experiment}_{dataset}.tif'
    write_tif(mosaic_file, mosaic, transform, crs)


def end_to_end_inference(root_dir: Path, cfg_file: Path, net_file: Path, city: str, tile_size: int = 256,
                             save_dir: Path = None, include_dataset: bool = False):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(cfg_file)
    net = load_net(cfg, net_file)

    # loading dataset from config (requires inference.json)
    ds = load_dataset(cfg, root_dir)
    year = ds.metadata['year']

    # getting width and height of mosaic by scanning all files
    files = [file for file in (root_dir / 'sentinel1').glob('**/*')]
    height, width = 0, 0
    for file in files:
        patch_product, patch_city, patch_year, patch_id = file.stem.split('_')
        if patch_city == city:
            m, n = patch_id.split('-')
            m, n = int(m), int(n)
            if m > height:
                height = m
            if n > width:
                width = n

    height += tile_size
    width += tile_size

    mosaic = np.empty((height, width, 1), dtype=np.uint8)
    mosaic_dataset = np.empty((height, width, 1), dtype=np.uint8)

    # classifying all tiles in dataset
    for i in range(len(ds)):

        sample_metadata = ds.metadata['samples'][i]
        patch_city = sample_metadata['city']
        print(patch_city)

        if patch_city == city:
            item = ds.__getitem__(i)
            img = item['x'].to(device)

            y_pred = net(img.unsqueeze(0))
            y_pred = torch.sigmoid(y_pred)

            y_pred = y_pred.cpu().detach().numpy()
            threshold = cfg.THRESH
            y_pred = y_pred[0, ] > threshold
            y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')

            patch_id = sample_metadata['patch_id']
            print(patch_id)
            m, n = patch_id.split('-')
            m, n = int(m), int(n)

            mosaic[m:m + tile_size, n:n + tile_size, 0] = y_pred[:, :, 0]
            if include_dataset:
                dataset = 1 if item['dataset'] == 'test' else 0
                mosaic_dataset[m:m + tile_size, n:n + tile_size, 0] = dataset

    # transform and crs from top left
    if save_dir is None:
        save_dir = root_dir / 'inference'
    if not save_dir.exists():
        save_dir.mkdir()

    # get transform and crs from top left patch
    m, n = 0, 0
    patch_id = f'{m:010d}-{n:010d}'
    patch_file = root_dir / 'sentinel1' / f'sentinel1_{city}_{year}_{patch_id}.tif'
    _, transform, crs = read_tif(patch_file)

    mosaic_file = save_dir / f'pred_{city}_{year}_{cfg_file.stem}.tif'
    write_tif(mosaic_file, mosaic, transform, crs)
    if include_dataset:
        dataset_file = save_dir / f'dataset_{city}_{year}_{cfg_file.stem}.tif'
        write_tif(dataset_file, mosaic_dataset, transform, crs)


def tile_inference(root_dir: Path, cfg_file: Path, net_file: Path, city: str, i_tile: str, j_tile: str,
                   tile_size: int = 256, save_dir: Path = None):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(cfg_file)
    net = load_net(cfg, net_file)

    # loading dataset from config (requires inference.json)
    ds = load_dataset(cfg, root_dir)
    year = ds.metadata['year']

    for i in range(len(ds)):
        sample_metadata = ds.metadata['samples'][i]
        patch_city = sample_metadata['city']
        patch_id = sample_metadata['patch_id']
        m, n = patch_id.split('-')

        if patch_city == city and (m == i_tile and n == j_tile):

            item = ds.__getitem__(i)
            img = item['x'].to(device)

            y_pred = net(img.unsqueeze(0))
            y_pred = torch.sigmoid(y_pred)

            y_pred = y_pred.cpu().detach().numpy()
            # threshold = cfg.THRESH
            threshold = float(net_file.stem.split('_')[-1]) / 100
            y_pred = y_pred[0, ] > threshold
            y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')

            if save_dir is None:
                save_dir = root_dir / 'inference'
            if not save_dir.exists():
                save_dir.mkdir()

            tile_file = save_dir / f'pred_{city}_{year}_{cfg_file.stem}_{patch_id}.tif'
            write_tif(tile_file, y_pred, item.get('transform'), item.get('crs'))
            return



if __name__ == '__main__':

    CFG_DIR = Path.cwd() / Path('configs/urban_extraction')
    NET_DIR = Path('/storage/shafner/run_logs/unet/')
    STORAGE_DIR = Path('/storage/shafner/urban_extraction')

    dataset = 'urban_extraction_stockholm_time_series'
    root_dir = STORAGE_DIR / dataset
    city = 'Stockholm'

    settings_single_scene = [
        ('10m_optical', 'net_9_41'),
        ('10m_fusion', 'net_15_38'),
        ('all_optical', 'net_9_47'),
        ('rgb', 'net_9_42'),
        ('no_rededge', 'net_9_44'),
        ('fusion_all', 'net_15_43'),
        ('VV_asc', 'net_7_27'),
        ('VH_asc', 'net_9_28'),
        ('dualpol_desc', 'net_9_32'),
        ('dualpol_dualorbit', 'net_7_30'),
        ('dualpol_asc', 'net_9_34')
    ]

    settings_time_series = [
        ('10m_optical_mean', 'net_6_32'),
        ('10m_optical_median', 'net_6_39'),
        ('10m_optical_allmetrics', 'net_6_38'),
        ('dualpol_asc_mean', 'net_6_35'),
        ('dualpol_asc_median', 'net_6_32'),
        ('dualpol_asc_allmetrics', 'net_6_43'),
        ('fusion_all_time_series', 'net_19_43')
    ]

    for setting in settings_time_series:
        print(setting)
        cfg, net = setting

        cfg_file = CFG_DIR / f'{cfg}.yaml'
        net_file = NET_DIR / cfg / f'{net}.pkl'

        tiles = [('0000003328', '0000007936'), ('0000003584', '0000009216'), ('0000004608', '0000008960'),
                 ('0000004352', '0000009216'), ('0000003840', '0000010496')]

        for tile in tiles:
            i, j = tile
            tile_inference(root_dir, cfg_file, net_file, city, i, j)

        # end_to_end_inference(root_dir, cfg_file, net_file, city, include_dataset=False)

