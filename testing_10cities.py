from pathlib import Path
import torch
from networks.network_loader import load_network
from experiment_manager.config import config
from utils.dataloader import InferenceDataset, UrbanExtractionDataset
from utils.geotiff import *
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')


def run_inference(config_name: str, checkpoint: int, site: str):
    # loading config and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = UrbanExtractionDataset(cfg, site, include_projection=True)

    # config inference directory
    save_path = ROOT_PATH / 'inference' / config_name
    save_path.mkdir(exist_ok=True)
    # site directory
    save_path = save_path / site
    save_path.mkdir(exist_ok=True)

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
            pred_file = save_path / f'pred_{config_name}_{patch_id}.tif'
            write_tif(pred_file, pred, transform, crs)

    combine_tif_patches(save_path, f'pred_{config_name}', delete_tiles=True)


if __name__ == '__main__':
    config_name = 'fusion'
    checkpoint = 50
    roi_id = 'milano'
    run_inference(config_name, checkpoint, roi_id)
