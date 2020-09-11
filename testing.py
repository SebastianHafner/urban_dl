from utils.visualization import *
from pathlib import Path
import numpy as np
import torch
from networks.network_loader import load_network
from utils.dataloader import SpaceNet7Dataset
from experiment_manager.config import config

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')


def qualitative_testing(config_name: str):
    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading dataset
    dataset = SpaceNet7Dataset(cfg)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    for index in range(len(dataset)):
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']

        with torch.no_grad():
            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0,])
            prob = prob.detach().cpu().numpy()
            pred = prob > cfg.THRESH

        fig, axs = plt.subplots(1, 3, figsize=(10, 6))

        # optical_file =
        # plot_optical(axs[0], optical_file, show_title=True)
        # plot_optical(axs[1], optical_file, vis='false_color', show_title=True)
        #
        # sar_file = DATASET_PATH / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        # plot_sar(axs[0, 2], sar_file, show_title=True)
        #
        # label = cfg.DATALOADER.LABEL
        # label_file = DATASET_PATH / site / label / f'{label}_{site}_{patch_id}.tif'
        # plot_buildings(axs[1, 0], label_file, show_title=True)



        plt.show()


if __name__ == '__main__':
    qualitative_testing('baseline_optical')
