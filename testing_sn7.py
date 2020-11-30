from utils.visualization import *
from pathlib import Path
import numpy as np
import json
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from networks.network_loader import load_network
from utils.datasets import SpaceNet7Dataset
from experiment_manager.config import config
from utils.metrics import *
from utils.geotiff import *


# TODO: add coordinates to title for area of interests

URBAN_EXTRACTION_PATH = Path('/storage/shafner/urban_extraction')
DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')

GROUPS = [(1, 'NA_AU', '#63cd93'), (2, 'SA', '#f0828f'), (3, 'EU', '#6faec9'), (4, 'SSA', '#5f4ad9'),
          (5, 'NAF_ME', '#8dee47'), (6, 'AS', '#d9b657'), ('total', 'Total', '#ffffff')]


def qualitative_testing(config_name: str, save_plots: bool = False):

    # loading dataset and networks
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = SpaceNet7Dataset(cfg)
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        group_index = int(sample['group']) - 1
        group = GROUPS[group_index][1]
        country = sample['country']

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

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
            pred = prob > cfg.INFERENCE.THRESHOLDS.VALIDATION

            plot_probability(axs[1, 2], prob, show_title=True)
            plot_prediction(axs[1, 1], pred, show_title=True)

        title = f'{config_name} {aoi_id} ({country}, {group})'
        if save_plots:
            path = URBAN_EXTRACTION_PATH / 'plots' / 'testing' / 'qualitative' / config_name
            path.mkdir(exist_ok=True)
            file = path / f'{title}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        else:
            fig.suptitle(title)
            plt.show()
        plt.close()


def quantitative_testing(config_name: str, save_output: bool = False):

    # loading config and dataset
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = SpaceNet7Dataset(cfg)
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    thresh = cfg.INFERENCE.THRESHOLDS.VALIDATION

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


def advanced_qualitative_testing(config_name: str, checkpoint: int, save_plots: bool = False):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading dataset
    dataset = SpaceNet7Dataset(cfg)

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    thresh = cfg.THRESHOLDS.VALIDATION

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        group_index = int(sample['group']) - 1
        group = GROUPS[group_index][1]
        country = sample['country']

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

        optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        plot_optical(axs[0, 0], optical_file, vis='false_color', show_title=True)

        sar_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        plot_sar(axs[0, 1], sar_file, show_title=True)

        label = cfg.DATALOADER.LABEL
        file_all = DATASET_PATH / 'sn7' / label / f'{label}_{aoi_id}.tif'
        ref_file = DATASET_PATH / 'sn7' / f'reference_{label}' / f'{label}_{aoi_id}.tif'
        # plot_stable_buildings(axs[0, 2], file_all, file_stable, show_title=True)
        plot_buildings(axs[0, 2], ref_file, show_title=True)

        with torch.no_grad():
            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0, ])
            prob = prob.detach().cpu().numpy()
            pred = prob > thresh

            plot_probability(axs[1, 0], prob, show_title=True)

            plot_probability_histogram(axs[1, 1], prob)

            # compute mean probabilities
            mean_prob_pos = np.mean(np.ma.array(prob, mask=np.logical_not(pred)))
            mean_prob_neg = np.mean(np.ma.array(1 - prob, mask=pred))
            axs[1, 1].set_title(f'hist prob ({mean_prob_pos:.2f} {mean_prob_neg:.2f})')

            plot_prediction(axs[1, 2], pred, show_title=True)


        title = f'{config_name} {aoi_id} ({country}, {group})'
        if save_plots:
            path = URBAN_EXTRACTION_PATH / 'plots' / 'testing' / 'histogram' / config_name
            path.mkdir(exist_ok=True)
            file = path / f'{title}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        else:
            fig.suptitle(title)
            plt.show()
        plt.close()


def out_of_distribution_check(n: int = 60, save_plots: bool = False):

    for index in range(n):

        fig, axs = plt.subplots(2, 5, figsize=(14, 6))

        for i, sensor in enumerate(['optical', 'sar']):

            # loading config and dataset
            cfg = config.load_cfg(CONFIG_PATH / f'{sensor}.yaml')
            dataset = SpaceNet7Dataset(cfg)

            sample = dataset.__getitem__(index)

            if i == 0:
                aoi_id = sample['aoi_id']
                group_index = int(sample['group']) - 1
                group = GROUPS[group_index][1]
                country = sample['country']

                optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
                plot_optical(axs[0, 0], optical_file, vis='false_color', show_title=True)

                sar_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
                plot_sar(axs[1, 0], sar_file, show_title=True)

            # loading network
            net = load_network(cfg, NETWORK_PATH / f'{sensor}_100.pkl')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(device)
            net.eval()
            thresh = cfg.THRESHOLDS.VALIDATION

            with torch.no_grad():
                x = sample['x'].to(device)
                logits = net(x.unsqueeze(0))
                prob = torch.sigmoid(logits.squeeze()).detach().cpu()
                pred = prob > thresh
                pred = pred
                gt = sample['y'].to('cpu').squeeze()

                f1 = f1_score(gt.flatten().unsqueeze(0), pred.flatten().unsqueeze(0)).item()

                prob, pred = prob.numpy(), pred.numpy()

                plot_probability(axs[i, 1], prob, show_title=True)

                plot_probability_histogram(axs[i, 2], prob)
                axs[i, 2].axvline(x=thresh, color='k')
                axs[i, 2].set_ylim((0, 10**5))

                # compute mean probabilities
                mean_prob_pos = np.mean(np.ma.array(prob, mask=np.logical_not(pred)))
                mean_prob_neg = np.mean(np.ma.array(1 - prob, mask=pred))
                axs[i, 2].set_title(f'hist prob ({mean_prob_pos:.2f} {mean_prob_neg:.2f})')

                plot_prediction(axs[i, 3], pred, show_title=True)
                axs[i, 3].set_title(f'pred F1 {f1:.3f}')

        label = cfg.DATALOADER.LABEL
        label_file = DATASET_PATH / 'sn7' / label / f'{label}_{aoi_id}.tif'
        plot_buildings(axs[0, 4], label_file, show_title=True)

        SN7_PATH = Path('/storage/shafner/spacenet7/train')
        change_file = SN7_PATH / aoi_id / 'auxiliary' / f'change.tif'
        arr, _, _ = read_tif(change_file)
        plot_stable_buildings_v2(axs[1, 4], arr)

        title = f'{aoi_id} ({country}, {group})'
        plt.suptitle(title)
        if save_plots:
            path = URBAN_EXTRACTION_PATH / 'plots' / 'testing' / 'out_of_distribution_detection'
            path.mkdir(exist_ok=True)
            file = path / f'out_of_distribution_detection_{aoi_id}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def out_of_distribution_correlation(config_name: str, checkpoint: int, save_plot: bool = False):

    # loading config and dataset
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = SpaceNet7Dataset(cfg)

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    thresh = cfg.THRESHOLDS.VALIDATION

    f1_scores = []
    mean_pos_probabilities = []
    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)

        with torch.no_grad():
            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits).cpu().detach().squeeze()
            pred = prob > thresh
            gt = sample['y'].to('cpu').detach().squeeze()
            f1 = f1_score(gt.flatten().unsqueeze(0), pred.flatten().unsqueeze(0)).item()
            f1_scores.append(f1)

            pred, prob = pred.numpy(), prob.numpy()

            # compute mean probabilities
            mean_prob_pos = np.mean(np.ma.array(prob, mask=np.logical_not(pred)))
            mean_pos_probabilities.append(mean_prob_pos)
            mean_prob_neg = np.mean(np.ma.array(1 - prob, mask=pred))

    # computing R square
    f1_scores = np.array(f1_scores)
    mean_pos_probabilities = np.array(mean_pos_probabilities)
    corr = np.corrcoef(mean_pos_probabilities, f1_scores)
    r_square = corr[0, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(mean_pos_probabilities, f1_scores)
    ax.set_xlim((0, 1))
    ax.set_xlabel('mean built-up probability')
    ax.set_ylim((0, 1))
    ax.set_ylabel('f1 score')
    title = f'correlation_{config_name}'
    ax.set_title(f'{title} (R square {r_square:.2f})')

    x = np.array([0, 1])
    m, b = np.polyfit(mean_pos_probabilities, f1_scores, 1)
    ax.plot(x, m * x + b, color='k')

    if save_plot:
        path = URBAN_EXTRACTION_PATH / 'plots' / 'testing' / 'out_of_distribution_detection' / config_name
        path.mkdir(exist_ok=True)
        file = path / f'{title}.png'
        plt.savefig(file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()



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


def qualitative_testing_comparison(config_names: list, checkpoints: list, save_plots: bool = False):

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
                pred = prob > cfg.THRESHOLDS.VALIDATION
                plot_prediction(axs[3 + i], pred, show_title=False)

        title = f'{aoi_id} ({country},  {group_name})'
        plt.suptitle(title)
        if save_plots:
            folder = URBAN_EXTRACTION_PATH / 'plots' / 'testing' / 'qualitative' / '_'.join(config_names)
            folder.mkdir(exist_ok=True)
            file = folder / f'{title}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_reference_comparison(start_index: int = 0):
    cfg = config.load_cfg(CONFIG_PATH / f'base_v3.yaml')
    dataset = SpaceNet7Dataset(cfg)

    for index in tqdm(range(len(dataset))):
        if index >= start_index:
            sample = dataset.__getitem__(index)
            aoi_id = sample['aoi_id']
            group_index = int(sample['group']) - 1
            group = GROUPS[group_index][1]
            country = sample['country']

            fig, axs = plt.subplots(2, 2, figsize=(6, 6))

            optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
            plot_optical(axs[0, 0], optical_file, vis='false_color', show_title=True)

            sar_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
            plot_sar(axs[0, 1], sar_file, show_title=True)

            label = cfg.DATALOADER.LABEL
            buildings_file = DATASET_PATH / 'sn7' / label / f'{label}_{aoi_id}.tif'
            plot_buildings(axs[1, 0], buildings_file, show_title=True)
            ref_buildings_file = DATASET_PATH / 'sn7' / f'reference_{label}' / f'{label}_{aoi_id}.tif'
            plot_buildings(axs[1, 1], ref_buildings_file, show_title=True)

            title = f'{aoi_id} ({country}, {group})'
            fig.suptitle(title)
            plt.show()
            plt.close()


if __name__ == '__main__':
    # qualitative_testing('sar', 100, save_plots=False)
    # advanced_qualitative_testing('fusion_color', 100, save_plots=False)
    # plot_reference_comparison(40)
    # out_of_distribution_check(60, save_plots=False)
    # out_of_distribution_correlation('optical', 100, save_plot=False)
    # quantitative_testing('optical_gamma_smallnet', 100, save_output=True)

    qualitative_testing('fusion_sensordropout', False)
    # quantitative_testing('sar_confidence', True)
    # quantitative_testing('sar_baseline_na', 100, save_output=True)

    # not including africa experiment
    # plot_quantitative_testing(['sar', 'sar_gamma_smallnet', 'optical', 'optical_gamma_smallnet'],
    #                           ['SAR', 'SAR gamma smallnet', 'optical', 'optical gamma smallnet'])

    # old vs. new
    # sar
    # plot_quantitative_testing(['baseline_sar', 'sar'], ['old sar', 'new sar'])
    # optical
    # plot_quantitative_testing(['baseline_optical', 'optical'], ['optical toa', 'optical sr'])
    # plot_quantitative_testing(['baseline_sar', 'sar', 'baseline_optical', 'optical'],
    #                           ['old sar', 'new sar', 'optical toa', 'optical sr'])

    # adding dsm to sar data experiment
    # plot_quantitative_testing(['baseline_sar', 'sar_dsm'], ['SAR', 'SAR with DSM'])

    # different fusions
    # plot_quantitative_testing(['baseline_fusion', 'sar_prediction_fusion', 'sar_prediction_dsm_fusion'],
    #                           ['sar + optical', 'sar pred + optical', 'sar pred + dsm + optical'])

    # plot_quantitative_testing(['baseline_sar', 'baseline_optical', 'baseline_fusion', 'sar_prediction_fusion'],
    #                           ['SAR', 'optical', 'fusion', 'new fusion'])

    # plot_quantitative_testing(['sar', 'optical', 'twostep_fusion'],
    #                           ['SAR', 'optical', 'twostep fusion'])

    # qualitative_testing_comparison(['baseline_sar', 'baseline_optical', 'baseline_fusion', 'sar_prediction_fusion'],
    #                                [100, 100, 100, 100], save_plots=True)

    # qualitative_testing_comparison(['fusion_gamma_smallnet', 'fusion_gamma_smallnet_sensordropout'], [100, 100], save_plots=False)

