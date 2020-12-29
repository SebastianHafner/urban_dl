from pathlib import Path
import torch
from networks.network_loader import load_network
from experiment_manager.config import config
from utils.datasets import GHSLDataset
from utils.geotiff import *
from tqdm import tqdm
import numpy as np
from utils.metrics import *

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def merge_patches(site: str, thresh: float = None):
    print(f'running inference for {site}...')

    dataset = GHSLDataset(ROOT_PATH / 'urban_dataset', site, thresh=thresh)
    patch_size = dataset.patch_size

    # config inference directory
    save_path = ROOT_PATH / 'inference' / 'ghsl'
    save_path.mkdir(exist_ok=True)

    ghsl_output = dataset.get_arr()
    transform, crs = dataset.get_geo()

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            ghsl_patch = patch['x'].cpu().squeeze().numpy()
            ghsl_patch = np.clip(ghsl_patch, 0, 100).astype('uint8')

            y, x = id2yx(patch['patch_id'])

            ghsl_output[y: y+patch_size, x:x+patch_size, 0] = ghsl_patch

    output_file = save_path / f'ghsl_{site}.tif'
    write_tif(output_file, ghsl_output, transform, crs)


def run_quantitative_evaluation(site: str, thresh: float = None):
    print(f'running inference for {site}...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GHSLDataset(ROOT_PATH / 'urban_dataset', site, thresh=thresh)

    y_preds, y_trues = None, None

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            ghsl_prob = patch['x'].to(device).squeeze().flatten()
            label = patch['y'].to(device)
            label = label.flatten().float()

            if y_preds is not None:
                y_preds = torch.cat((y_preds, ghsl_prob), dim=0)
                y_trues = torch.cat((y_trues, label), dim=0)
            else:
                y_preds = ghsl_prob
                y_trues = label

        prec = precision(y_trues, y_preds, dim=0)
        rec = recall(y_trues, y_preds, dim=0)
        f1 = f1_score(y_trues, y_preds, dim=0)
        print(f'{site}: f1 score {f1:.3f} - precision {prec:.3f} - recall {rec:.3f}')




if __name__ == '__main__':
    cities_igarss = ['stockholm', 'kampala', 'daressalam', 'sidney']
    for city in cities_igarss:
        # merge_patches(city)
        run_quantitative_evaluation(city, thresh=50)
