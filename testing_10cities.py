from pathlib import Path
import torch
from networks.network_loader import load_network
from experiment_manager.config import config
from utils.dataloader import InferenceDataset, UrbanExtractionDataset
from utils.geotiff import *
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def run_inference(config_name: str, site: str):

    print(f'running inference for {site}...')

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = UrbanExtractionDataset(cfg, site, no_augmentations=True, include_projection=True)

    # config inference directory
    save_path = ROOT_PATH / 'inference' / config_name
    save_path.mkdir(exist_ok=True)
    # temporary directory
    temp_path = save_path / f'temp_{site}'
    temp_path.mkdir(exist_ok=True)

    basename = f'pred_{site}_{config_name}'

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            img = patch['x'].to(device)
            logits = net(img.unsqueeze(0))
            prob = torch.sigmoid(logits)
            pred = prob > cfg.THRESHOLDS.VALIDATION
            pred = pred.squeeze().cpu().numpy().astype('uint8')

            transform = patch['transform']
            crs = patch['crs']
            patch_id = patch['patch_id']
            pred_file = temp_path / f'{basename}_{patch_id}.tif'
            write_tif(pred_file, pred, transform, crs)

    combine_tif_patches(temp_path, basename, delete_tiles=True)
    shutil.move(temp_path / f'{basename}.tif', save_path / f'{basename}.tif')
    temp_path.rmdir()


def patches2png(site: str, sensor: str, band_indices: list, rescale_factor: float = 1, patch_size: int = 256):

    # config inference directory
    save_path = ROOT_PATH / 'plots' / 'inspection'
    save_path.mkdir(exist_ok=True)

    folder = ROOT_PATH / 'urban_extraction_dataset' / site / sensor
    files = [f for f in folder.glob('**/*')]

    files_per_side = int(np.ceil(np.sqrt(len(files))))
    arr = np.zeros((files_per_side * patch_size, files_per_side * patch_size, 3))
    for index, f in enumerate(tqdm(files)):
        patch, _, _ = read_tif(f)
        m, n, _ = patch.shape
        i = (index // files_per_side) * patch_size
        j = (index % files_per_side) * patch_size
        if len(band_indices) == 3:
            arr[i:i+m, j:j+n, ] = patch[:, :, band_indices]
        else:
            for b in range(3):
                arr[i:i + patch_size, j:j + patch_size, b] = patch[:, :, band_indices]

    plt.imshow(np.clip(arr / rescale_factor, 0, 1))
    save_file = save_path / f'{site}_{sensor}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    config_name = 'sar'
    cities = ['stockholm', 'beijing', 'jakarta', 'kigali', 'lagos', 'mexicocity', 'milano', 'mumbai', 'riodejanairo',
              'sidney']
    for city in cities:
        patches2png(city, 'sentinel2', [2, 1, 0], 0.4)
        # run_inference(config_name, city)
