from utils.visualization import *
from pathlib import Path
import numpy as np
import json
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from networks.network_loader import load_network
from utils.dataloader import SpaceNet7Dataset
from experiment_manager.config import config
from utils.metrics import *
from utils.geotiff import *

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')

GROUPS = [(1, 'NA_AU', '#63cd93'), (2, 'SA', '#f0828f'), (3, 'EU', '#6faec9'), (4, 'SSA', '#5f4ad9'),
          (5, 'NAF_ME', '#8dee47'), (6, 'AS', '#d9b657'), ('total', 'Total', '#ffffff')]


def qualitative_testing(config_name: str, checkpoint: int, threshold: int = None):
    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading dataset
    dataset = SpaceNet7Dataset(cfg)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    thresh = cfg.THRESH if threshold is None else threshold

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle(aoi_id)

        optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        plot_optical(axs[0, 0], optical_file, show_title=True)
        plot_optical(axs[0, 1], optical_file, vis='false_color', show_title=True)

        sar_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        plot_sar(axs[0, 2], sar_file, show_title=True)


        label = cfg.DATALOADER.LABEL
        label_file = DATASET_PATH / 'sn7' / label / f'{label}_{aoi_id}.tif'
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


def quantitative_testing(config_name: str, checkpoint: int, save_output: bool = False, threshold: float = None):
    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading dataset
    dataset = SpaceNet7Dataset(cfg)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    thresh = cfg.THRESHOLDS.VALIDATION

    y_true_dict = {'total': np.array([])}
    y_pred_dict = {'total': np.array([])}
    # container for output file
    output_data = {
        'metadata': {'cfg_name': config_name, 'thresh': thresh},
        'groups': GROUPS,
        'data': {}
    }

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)
        group = sample['group']

        with torch.no_grad():
            x = sample['x'].to(device)
            y_true = sample['y'].to(device)
            logits = net(x.unsqueeze(0))
            y_pred = torch.sigmoid(logits) > thresh

            y_true = y_true.detach().cpu().flatten().numpy()
            y_pred = y_pred.detach().cpu().flatten().numpy()

            if group not in y_true_dict.keys():
                y_true_dict[group] = y_true
                y_pred_dict[group] = y_pred
            else:
                y_true_dict[group] = np.concatenate((y_true_dict[group], y_true))
                y_pred_dict[group] = np.concatenate((y_pred_dict[group], y_pred))

            y_true_dict['total'] = np.concatenate((y_true_dict['total'], y_true))
            y_pred_dict['total'] = np.concatenate((y_pred_dict['total'], y_pred))

    for group in GROUPS:
        group_index, group_name, _ = group
        group_y_true = torch.Tensor(np.array(y_true_dict[group_index]))
        group_y_pred = torch.Tensor(np.array(y_pred_dict[group_index]))
        prec = precision(group_y_true, group_y_pred, dim=0).item()
        rec = recall(group_y_true, group_y_pred, dim=0).item()
        f1 = f1_score(group_y_true, group_y_pred, dim=0).item()

        output_data['data'][group_index] = {'f1_score': f1, 'precision': prec, 'recall': rec}
        print(f'{group_name} ({group_index}) - Precision: {prec:.3f} - Recall: {rec:.3f} - F1 score: {f1:.3f}')

    if save_output:
        output_file = DATASET_PATH.parent / 'testing' / f'testing_{config_name}.json'
        with open(str(output_file), 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)


def plot_quantitative_testing(config_names: list, names: list):

    path = DATASET_PATH.parent / 'testing'
    data = [load_json(path / f'testing_{config_name}.json') for config_name in config_names]

    metrics = ['f1_score', 'precision', 'recall']
    groups = data[0]['groups']
    group_names = [group[1] for group in groups]

    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(10, 4))
        width = 0.2
        for j, experiment in enumerate(data):
            ind = np.arange(len(groups))
            y_pos = [experiment['data'][str(group[0])][metric] for group in groups]
            x_pos = ind + (j * width)
            ax.bar(x_pos, y_pos, width, label=names[j], zorder=3, edgecolor='black')

        ax.set_ylim((0, 1))
        ax.set_ylabel(metric)
        ax.legend(loc='best')
        x_ticks = ind + (len(config_names) - 1) * width / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(group_names)
        plt.grid(b=True, which='major', axis='y', zorder=0)
        plt.show()


def qualitative_testing_comparison(config_names: list, checkpoints: list):

    # setup
    configs = [config.load_cfg(CONFIG_PATH / f'{config_name}.yaml') for config_name in config_names]
    datasets = [SpaceNet7Dataset(cfg) for cfg in configs]
    net_files = [NETWORK_PATH / f'{name}_{checkpoint}.pkl' for name, checkpoint in zip(config_names, checkpoints)]

    # optical, sar, reference and predictions (n configs)
    n_plots = 3 + len(config_names)

    for index in range(len(datasets[0])):
        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))
        for i, (cfg, dataset, net_file) in enumerate(zip(configs, datasets, net_files)):

            net = load_network(cfg, net_file)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(device)
            net.eval()

            sample = dataset.__getitem__(index)
            aoi_id = sample['aoi_id']
            country = sample['country']
            group_name = sample['group_name']

            if i == 0:
                super_title = f'{aoi_id} ({country},  {group_name})'
                fig.suptitle(super_title, size=20)
                fig.subplots_adjust(wspace=0, hspace=0)
                mpl.rcParams['axes.linewidth'] = 4

                optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
                plot_optical(axs[0], optical_file, vis='false_color', show_title=False)

                sar_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
                plot_sar(axs[1], sar_file, show_title=False)

                label = cfg.DATALOADER.LABEL
                label_file = DATASET_PATH / 'sn7' / label / f'{label}_{aoi_id}.tif'
                plot_buildings(axs[2], label_file, show_title=False)

            with torch.no_grad():
                x = sample['x'].to(device)
                logits = net(x.unsqueeze(0))
                prob = torch.sigmoid(logits[0, 0,])
                prob = prob.detach().cpu().numpy()
                pred = prob > cfg.THRESH
                plot_prediction(axs[3 + i], pred, show_title=False)

        plt.show()


if __name__ == '__main__':
    # qualitative_testing('sar_dsm', 100)
    # quantitative_testing('twostep_fusion', 100, save_output=True)
    # quantitative_testing('optical_baseline_na', 100, save_output=True)
    # quantitative_testing('sar_baseline_na', 100, save_output=True)

    # not including africa experiment
    # plot_quantitative_testing(['baseline_sar', 'sar_baseline_na', 'baseline_optical', 'optical_baseline_na'],
    #                           ['SAR', 'SAR na', 'optical', 'optical na'])

    # adding dsm to sar data experiment
    # plot_quantitative_testing(['baseline_sar', 'sar_dsm'], ['SAR', 'SAR with DSM'])

    # different fusions
    # plot_quantitative_testing(['baseline_fusion', 'sar_prediction_fusion', 'sar_prediction_dsm_fusion'],
    #                           ['sar + optical', 'sar pred + optical', 'sar pred + dsm + optical'])


    # plot_quantitative_testing(['baseline_sar', 'baseline_optical', 'baseline_fusion', 'sar_prediction_fusion'],
    #                           ['SAR', 'optical', 'fusion', 'new fusion'])

    plot_quantitative_testing(['sar', 'optical', 'twostep_fusion'],
                              ['SAR', 'optical', 'twostep fusion'])

    # qualitative_testing_comparison(['baseline_sar', 'baseline_optical', 'baseline_fusion', 'sar_prediction_fusion'],
    #                                [100, 100, 100, 100])
    pass
