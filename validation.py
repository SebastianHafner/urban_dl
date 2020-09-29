from pathlib import Path
from networks.network_loader import load_network
from utils.dataloader import UrbanExtractionDataset
from experiment_manager.config import config
from utils.metrics import *
from tqdm import tqdm
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from utils.visualization import *

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')


def quantitative_validation(config_name: str, checkpoint: int, save_output: bool = False):

    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    thresh = cfg.THRESHOLDS.TRAIN

    output_data = {'training': {}, 'validation': {}}
    for run_type in ['training', 'validation']:
        sites = cfg.DATASETS.SITES.TRAINING if run_type == 'training' else cfg.DATASETS.SITES.VALIDATION
        for site in sites:
            print(f'Quantitative assessment {site} ({run_type})')
            dataset = UrbanExtractionDataset(cfg=cfg, dataset=site, no_augmentations=True)
            y_true_set, y_pred_set = np.array([]), np.array([])
            for index in tqdm(range(len(dataset))):
                sample = dataset.__getitem__(index)

                with torch.no_grad():
                    x = sample['x'].to(device)
                    y_true = sample['y'].to(device)
                    logits = net(x.unsqueeze(0))
                    y_pred = torch.sigmoid(logits) > thresh

                    y_true = y_true.detach().cpu().flatten().numpy()
                    y_pred = y_pred.detach().cpu().flatten().numpy()
                    y_true_set = np.concatenate((y_true_set, y_true))
                    y_pred_set = np.concatenate((y_pred_set, y_pred))

            y_true_set, y_pred_set = torch.Tensor(np.array(y_true_set)), torch.Tensor(np.array(y_pred_set))
            prec = precision(y_true_set, y_pred_set, dim=0)
            rec = recall(y_true_set, y_pred_set, dim=0)
            f1 = f1_score(y_true_set, y_pred_set, dim=0)

            print(f'Precision: {prec.item():.3f} - Recall: {rec.item():.3f} - F1 score: {f1.item():.3f}')

            output_data[run_type][site] = {'f1_score': f1.item(), 'precision': prec.item(), 'recall': rec.item()}

    if save_output:
        output_file = DATASET_PATH.parent / 'validation' / f'validation_{config_name}.json'
        with open(str(output_file), 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)


def plot_quantitative_validation(config_names: list, names: list, run_type: str):
    def load_data(config_name: str):
        file = DATASET_PATH.parent / 'validation' / f'validation_{config_name}.json'
        d = load_json(file)
        return d[run_type]

    data = [load_data(config_name) for config_name in config_names]
    width = 0.2

    metrics = ['f1_score', 'precision', 'recall']
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(10, 4))
        for j, experiment in enumerate(data):
            sites = experiment.keys()
            ind = np.arange(len(sites))
            experiment_data = [experiment[site][metric] for site in sites]
            ax.bar(ind + (j * width), experiment_data, width, label=names[j], zorder=3)

        ax.set_ylim((0, 1))
        ax.set_ylabel(metric)
        ax.legend(loc='best')
        ax.set_xticks(ind)
        ax.set_xticklabels(sites)
        plt.grid(b=True, which='major', axis='y', zorder=0)
        plt.show()


def random_selection(config_name: str, site: str, n: int):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading dataset
    dataset = UrbanExtractionDataset(cfg, site, no_augmentations=True)

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    thresh = cfg.THRESHOLD.TRAIN

    item_indices = list(np.random.randint(0, len(dataset), size=n))

    for index in item_indices:
        sample = dataset.__getitem__(index)
        patch_id = sample['patch_id']

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

        optical_file = DATASET_PATH / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        plot_optical(axs[0, 0], optical_file, show_title=True)
        plot_optical(axs[0, 1], optical_file, vis='false_color', show_title=True)

        sar_file = DATASET_PATH / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        plot_sar(axs[0, 2], sar_file, show_title=True)

        label = cfg.DATALOADER.LABEL
        label_file = DATASET_PATH / site / label / f'{label}_{site}_{patch_id}.tif'
        plot_buildings(axs[1, 0], label_file, show_title=True)

        with torch.no_grad():
            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0,])
            prob = prob.detach().cpu().numpy()
            pred = prob > thresh

            plot_activation(axs[1, 2], prob, show_title=True)
            plot_prediction(axs[1, 1], pred, show_title=True)

        plt.show()


if __name__ == '__main__':
    config_name = 'sar'
    checkpoint = 100

    # quantitative_validation(config_name, checkpoint, True)
    plot_quantitative_validation(['sar'], ['sar'], run_type='validation')
    # random_selection(config_name, 'montreal', 20)

    # fusion_file = DATASET_PATH.parent / 'quantitative_assessment' / f'qantitative_assessment_baseline_fusion.json'
    # optical_file = DATASET_PATH.parent / 'quantitative_assessment' / f'qantitative_assessment_baseline_optical.json'
    # sar_file = DATASET_PATH.parent / 'quantitative_assessment' / f'qantitative_assessment_baseline_sar.json'
    # plot_quantitative_results([sar_file, optical_file, fusion_file], ['SAR', 'optical', 'fusion'], 'validation')



