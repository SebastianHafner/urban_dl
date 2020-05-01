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
import torchvision.transforms.functional as TF
from torch.utils import data as torch_data

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np


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
    dataset = dataloader.UrbanExtractionDataset(cfg, root_dir, 'test', include_projection=True)
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

    # loading cfg, network and dataset (requires inference.json)
    cfg = load_cfg(cfg_file)
    net = load_net(cfg, net_file)
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


def evaluate_patches_new(root_dir: Path, cfg_file: Path, net_file: Path, dataset: str = 'test', n: int = 10,
                     save_dir: Path = None):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(cfg_file)
    net = load_net(cfg, net_file)

    # loading dataset from config (requires inference.json)
    ds = dataloader.UrbanExtractionDataset(cfg, dataset, include_projection=True)

    np.random.seed(7)
    indices = np.random.randint(1, len(ds), n)

    for index in list(indices):

        # getting item
        item = ds.__getitem__(index)
        city = item['city']
        patch_id = item['patch_id']
        x = item['x'].to(device)

        # network prediction
        y_pred = net(x.unsqueeze(0))
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = y_pred[0, ] > cfg.THRESH
        y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')

        # label
        y = item['y'].to(device)
        y = y.cpu().detach().numpy()
        y = y.transpose((1, 2, 0)).astype('uint8')

        # sentinel data
        x = x.cpu().detach().numpy()
        x = x.transpose((1, 2, 0))
        rgb = x[:, :, [4, 3, 2]] / 0.3
        rgb = np.minimum(rgb, 1)
        vv = x[:, :, 0]

        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for ax in axs:
            ax.set_axis_off()

        axs[0].imshow(y[:, :, 0])
        axs[1].imshow(y_pred[:, :, 0])
        axs[2].imshow(rgb)
        axs[3].imshow(vv, cmap='gray')

        if save_dir is None:
            save_dir = root_dir / 'evaluation' / cfg_file.stem
        if not save_dir.exists():
            save_dir.mkdir()
        file = save_dir / f'eval_{cfg_file.stem}_{city}_{patch_id}.png'

        plt.savefig(file, dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_patches(root_dir: Path, cfg_file: Path, net_file: Path, city: str, dataset: str = 'test',
                     save_dir: Path = None):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(cfg_file)
    net = load_net(cfg, net_file)

    # loading dataset from config (requires inference.json)
    ds = dataloader.UrbanExtractionDataset(cfg, root_dir, dataset, include_projection=True)
    samples = ds.metadata['samples']
    samples = sorted(samples, key=lambda sample: sample['img_weight'], reverse=True)

    top = 10
    for i in range(top):
        patch_id = samples[i]['patch_id']
        index = ds.get_index(city, patch_id)

        pred, _, _ = ds.classify_item(index, net, device)
        label, _, _ = ds._get_label_data(city, patch_id)

        false_positives = np.logical_and(pred == 1,  label == 0)[:, :, 0]
        false_negatives = np.logical_and(pred == 0, label == 1)[:, :, 0]
        true_positives = np.logical_and(pred, label)[:, :, 0]

        eval = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        eval[true_positives, :] = [255, 255, 255]
        eval[false_positives, :] = [0, 0, 255]
        eval[false_negatives, :] = [255, 0, 0]



        # img = np.repeat(pred * 255, 3, axis=2)
        if save_dir is None:
            save_dir = root_dir / 'evaluation' / cfg_file.stem
        if not save_dir.exists():
            save_dir.mkdir()
        file = save_dir / f'eval_{city}_{patch_id}.png'
        image.imsave(file, eval)




        # patch_id = f'{i_patch:010d}-{j_patch:010d}'
        # index = ds.get_index_from_patch_id(patch_id)

def img2map(cfg_file: Path, net_file: Path, s1_file: Path, s2_file: Path, save_dir: Path):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(cfg_file)
    net = load_net(cfg, net_file)

    # loading dataset from config (requires inference.json)
    patch_size = 256
    ds = dataloader.InferenceDataset(cfg, s1_file=s1_file, s2_file=s2_file, patch_size=patch_size)
    pred = np.empty(shape=(ds.n_rows * ds.patch_size, ds.n_cols * ds.patch_size, 1), dtype='uint8')
    activation = np.empty(shape=(ds.n_rows * ds.patch_size, ds.n_cols * ds.patch_size, 1), dtype='float32')

    for i in range(len(ds)):
        print(f'patch {i + 1}/{len(ds)}')
        patch = ds.__getitem__(i)

        img_patch = patch['x'].to(device)
        i_start, i_end = patch['row']
        j_start, j_end = patch['col']

        activation_patch = net(img_patch.unsqueeze(0))
        activation_patch = torch.sigmoid(activation_patch)
        activation_patch = activation_patch.cpu().detach().numpy()
        activation_patch = activation_patch[0, ].transpose((1, 2, 0)).astype('float32')
        pred_patch = activation_patch > cfg.THRESH
        pred_patch = pred_patch.astype('uint8')
        rf = ds.rf
        if i_start == 0 and j_start == 0:
            activation[i_start:i_end-2*rf, j_start:j_end-2*rf, ] = activation_patch[:-2*rf, :-2*rf, ]
            pred[i_start:i_end - 2 * rf, j_start:j_end - 2 * rf, ] = pred_patch[:-2 * rf, :-2 * rf, ]
        elif i_start == 0:
            activation[i_start:i_end-2*rf, j_start+rf:j_end-rf, ] = activation_patch[:-2*rf, rf:-rf, ]
            pred[i_start:i_end - 2 * rf, j_start + rf:j_end - rf, ] = pred_patch[:-2 * rf, rf:-rf, ]
        elif j_start == 0:
            activation[i_start+rf: i_end-rf, j_start: j_end-2*rf, ] = activation_patch[rf:-rf, :-2*rf, ]
            pred[i_start + rf: i_end - rf, j_start: j_end - 2 * rf, ] = pred_patch[rf:-rf, :-2 * rf, ]
        else:
            activation[i_start+rf: i_end-rf, j_start+rf: j_end-rf, ] = activation_patch[rf:-rf, rf:-rf, ]
            pred[i_start + rf: i_end - rf, j_start + rf: j_end - rf, ] = pred_patch[rf:-rf, rf:-rf, ]

    save_dir.mkdir(exist_ok=True)
    activation_file = save_dir / f'activation_{cfg_file.stem}_stockholm.tif'
    write_tif(activation_file, activation, ds.geotransform, ds.crs)
    pred_file = save_dir / f'pred_{cfg_file.stem}_stockholm.tif'
    write_tif(pred_file, pred, ds.geotransform, ds.crs)


if __name__ == '__main__':

    CFG_DIR = Path.cwd() / Path('configs/urban_extraction')
    NET_DIR = Path('/storage/shafner/run_logs/unet/')
    STORAGE_DIR = Path('/storage/shafner/urban_extraction')

    dataset = 'urban_extraction_buildings'
    root_dir = STORAGE_DIR / dataset
    cfg = 'benchmark_africa'

    cfg_file = CFG_DIR / f'{cfg}.yaml'
    net_file = NET_DIR / cfg / 'best_net.pkl'

    # evaluate_patches_new(root_dir, cfg_file, net_file, 'test', 100)

    s1_file = STORAGE_DIR / dataset / 'inference' / 'data' / 'sentinel1_stockholm.tif'
    s2_file = STORAGE_DIR / dataset / 'inference' / 'data' / 'sentinel2_stockholm.tif'

    save_dir = STORAGE_DIR / dataset / 'inference' / cfg
    img2map(cfg_file, net_file, s1_file, s2_file, save_dir)

