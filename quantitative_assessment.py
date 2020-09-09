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

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')


def compute_accuracy_metrics(config_name: str, output_file: Path):

    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    data = {'training': {}, 'validation': {}}
    for run_type in ['training', 'validation']:
        sites = cfg.DATASETS.SITES.TRAINING if run_type == 'training' else cfg.DATASETS.SITES.VALIDATION
        for site in sites:
            print(f'Quantitative assessment {site} ({run_type})')
            dataset = UrbanExtractionDataset(cfg=cfg, dataset=site, no_augmentations=True)
            y_true_set = np.array([])
            y_pred_set = np.array([])

            for index in tqdm(range(len(dataset))):
                sample = dataset.__getitem__(index)

                with torch.no_grad():
                    x = sample['x'].to(device)
                    y_true = sample['y'].to(device)
                    logits = net(x.unsqueeze(0))
                    y_pred = torch.sigmoid(logits) > cfg.THRESH

                    y_true = y_true.detach().cpu().flatten().numpy()
                    y_pred = y_pred.detach().cpu().flatten().numpy()
                    y_true_set = np.concatenate((y_true_set, y_true))
                    y_pred_set = np.concatenate((y_pred_set, y_pred))

            y_true_set, y_pred_set = torch.Tensor(np.array(y_true_set)), torch.Tensor(np.array(y_pred_set))
            prec = precision(y_true_set, y_pred_set, dim=0)
            rec = recall(y_true_set, y_pred_set, dim=0)
            f1 = f1_score(y_true_set, y_pred_set, dim=0)

            print(f'Precision: {prec.item():.3f} - Recall: {rec.item():.3f} - F1 score: {f1.item():.3f}')

            data[run_type][site] = {'f1_score': f1.item(), 'precision': prec.item(), 'recall': rec.item()}

    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def plot_quantitative_results(files: list, names: list, run_type: str):

    def load_data(file: Path):
        with open(str(file)) as f:
            d = json.load(f)
        return d[run_type]

    data = [load_data(file) for file in files]
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


if __name__ == '__main__':
    config_name = 'baseline_fusion'
    output_file = DATASET_PATH.parent / 'quantitative_assessment' / f'qantitative_assessment_{config_name}.json'
    # compute_accuracy_metrics(config_name, output_file)


    fusion_file = DATASET_PATH.parent / 'quantitative_assessment' / f'qantitative_assessment_baseline_fusion.json'
    optical_file = DATASET_PATH.parent / 'quantitative_assessment' / f'qantitative_assessment_baseline_optical.json'
    sar_file = DATASET_PATH.parent / 'quantitative_assessment' / f'qantitative_assessment_baseline_sar.json'
    plot_quantitative_results([sar_file, optical_file, fusion_file], ['SAR', 'optical', 'fusion'], 'validation')
