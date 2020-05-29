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


def evaluate_patches(root_dir: Path, cfg_file: Path, net_file: Path, dataset: str = 'test', n: int = 10,
                     save_dir: Path = None, label_pred_only: bool = False):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading cfg and network
    cfg = load_cfg(cfg_file)
    net = load_net(cfg, net_file)

    # loading dataset from config (requires inference.json)
    ds = dataloader.UrbanExtractionDataset(cfg, dataset, include_projection=True)

    np.random.seed(7)
    indices = np.random.randint(1, len(ds), n)

    for i, index in enumerate(indices):
        print(f'{i}/{n}')
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

        if label_pred_only:
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            axs[0].imshow(y[:, :, 0])
            axs[1].imshow(y_pred[:, :, 0])

        else:
            # sentinel data
            x = x.cpu().detach().numpy()
            x = x.transpose((1, 2, 0))
            rgb = x[:, :, [4, 3, 2]] / 0.3
            rgb = np.minimum(rgb, 1)
            vv = x[:, :, 0]

            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            axs[0].imshow(y[:, :, 0])
            axs[1].imshow(y_pred[:, :, 0])
            axs[2].imshow(rgb)
            axs[3].imshow(vv, cmap='gray')

        for ax in axs:
            ax.set_axis_off()

        if save_dir is None:
            save_dir = root_dir / 'evaluation' / cfg_file.stem
        if not save_dir.exists():
            save_dir.mkdir()
        file = save_dir / f'eval_{cfg_file.stem}_{city}_{patch_id}.png'

        plt.savefig(file, dpi=300, bbox_inches='tight')
        plt.close()


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
    cfg = 'sensor_dropout'

    cfg_file = CFG_DIR / f'{cfg}.yaml'
    net_file = NET_DIR / cfg / 'best_net.pkl'

    # evaluate_patches(root_dir, cfg_file, net_file, 'test', 100, label_pred_only=True)

    s1_file = STORAGE_DIR / dataset / 'inference' / 'data' / 'sentinel1_stockholm.tif'
    s2_file = STORAGE_DIR / dataset / 'inference' / 'data' / 'sentinel2_stockholm.tif'

    save_dir = STORAGE_DIR / dataset / 'inference' / cfg
    img2map(cfg_file, net_file, s1_file, s2_file, save_dir)

