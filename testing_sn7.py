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
from sklearn.metrics import precision_recall_curve


URBAN_EXTRACTION_PATH = Path('/storage/shafner/urban_extraction')
ROOT_PATH = Path('/storage/shafner/urban_extraction')
DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')
ROOT_PATH = Path('/storage/shafner/urban_extraction')

# TODO: add this to dataset
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


def quantitative_testing(config_name: str, threshold: float = None, save_output: bool = False):

    # loading config and dataset
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = SpaceNet7Dataset(cfg)
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    thresh = threshold if threshold else cfg.INFERENCE.THRESHOLDS.VALIDATION

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

    # computing R squarec
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


def show_quantitative_testing(config_name: str):

    data = load_json(DATASET_PATH.parent / 'testing' / f'testing_{config_name}.json')
    print(config_name)
    for metric in ['f1_score', 'precision', 'recall']:
        print(metric)
        group_values = []
        for group in data['groups']:
            group_index = group[0]
            group_name = group[1]
            value = data['data'][str(group_index)][metric]
            if group != 'total':
                group_values.append(value)
            print(f'{group_name}: {value:.3f},', end=' ')
        print('')
        min_ = np.min(group_values)
        max_ = np.max(group_values)
        mean = np.mean(group_values)
        std = np.std(group_values)
        print(f'summary statistics: {min_:.3f} min, {max_:.3f} max, {mean:.3f} mean, {std:.3f} std')



def plot_quantitative_testing(config_names: list, names: list):

    mpl.rcParams.update({'font.size': 20})
    path = DATASET_PATH.parent / 'testing'
    data = [load_json(path / f'testing_{config_name}.json') for config_name in config_names]

    metrics = ['f1_score', 'precision', 'recall']
    metric_names = ['F1 score', 'Precision', 'Recall']
    groups = data[0]['groups'][:-1]
    group_names = [group[1] for group in groups]

    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.2
        for j, experiment in enumerate(data):
            ind = np.arange(len(groups))
            y_pos = [experiment['data'][str(group[0])][metric] for group in groups]
            x_pos = ind + (j * width)
            ax.bar(x_pos, y_pos, width, label=names[j], zorder=3, edgecolor='black')

        ax.set_ylim((0, 1))
        ax.set_ylabel(metric_names[i])
        if i == 0:
            ax.legend(loc='upper center', ncol=4, frameon=False, handletextpad=0.8, columnspacing=1, handlelength=1)
        x_ticks = ind + (len(config_names) - 1) * width / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(group_names)
        plt.grid(b=True, which='major', axis='y', zorder=0)
        plt.show()


def plot_activation_comparison(config_names: list, save_plots: bool = False):

    # setup
    configs = [config.load_cfg(CONFIG_PATH / f'{config_name}.yaml') for config_name in config_names]
    datasets = [SpaceNet7Dataset(cfg) for cfg in configs]
    net_files = [NETWORK_PATH / f'{name}_{cfg.INFERENCE.CHECKPOINT}.pkl' for cfg, name in zip(configs, config_names)]

    # optical, sar, reference and predictions (n configs)
    n_plots = 3 + len(config_names)

    for index in range(len(datasets[0])):
        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 3, 4))
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
                mpl.rcParams['axes.linewidth'] = 1

                optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
                plot_optical(axs[0], optical_file, vis='true_color', show_title=False)

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
                plot_probability(axs[3 + i], prob)

        title = f'{country} ({group_name})'

        # plt.suptitle(title, ha='left', va='center', x=0, y=0.5, fontsize=10, rotation=90)
        axs[0].set_ylabel(title, fontsize=16)
        if save_plots:
            folder = URBAN_EXTRACTION_PATH / 'plots' / 'testing' / 'qualitative' / '_'.join(config_names)
            folder.mkdir(exist_ok=True)
            file = folder / f'{aoi_id}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_activation_comparison_assembled(config_names: list, names: list, aoi_ids: list = None,
                                         save_plot: bool = False):
    mpl.rcParams['axes.linewidth'] = 1
    fontsize = 18

    # setting up plot
    plot_size = 3
    plot_rows = len(aoi_ids)
    plot_height = plot_size * plot_rows
    plot_cols = 3 + len(config_names)  # optical, sar, reference and predictions (n configs)
    plot_width = plot_size * plot_cols
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(plot_width, plot_height))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, config_name in enumerate(config_names):

        # loading configs, datasets, and networks
        cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
        dataset = SpaceNet7Dataset(cfg)
        net = load_network(cfg, NETWORK_PATH / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        net.eval()

        for j, aoi_id in enumerate(aoi_ids):
            index = dataset.get_index(aoi_id)
            sample = dataset.__getitem__(index)
            country = sample['country']
            if country == 'United States':
                country = 'US'
            if country == 'United Kingdom':
                country = 'UK'
            if country == 'Saudi Arabia':
                country = 'Saudi Ar.'
            group_name = sample['group_name']

            optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
            plot_optical(axs[j, 0], optical_file, vis='true_color', show_title=False)
            axs[-1, 0].set_xlabel('(a) Image Optical', fontsize=fontsize)

            sar_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
            plot_sar(axs[j, 1], sar_file, show_title=False)
            axs[-1, 1].set_xlabel('(b) Image SAR', fontsize=fontsize)

            label = cfg.DATALOADER.LABEL
            label_file = DATASET_PATH / 'sn7' / label / f'{label}_{aoi_id}.tif'
            plot_buildings(axs[j, 2], label_file, show_title=False)
            axs[-1, 2].set_xlabel('(c) Ground Truth', fontsize=fontsize)

            with torch.no_grad():
                x = sample['x'].to(device)
                logits = net(x.unsqueeze(0))
                prob = torch.sigmoid(logits.squeeze())
                prob = prob.cpu().numpy()
                ax = axs[j, 3 + i]
                plot_probability(ax, prob)

            if i == 0:  # row labels only need to be set once
                row_label = f'{country} ({group_name})'
                axs[j, 0].set_ylabel(row_label, fontsize=fontsize)

            col_letter = chr(ord('a') + 3 + i)
            col_label = f'({col_letter}) {names[i]}'
            axs[-1, 3 + i].set_xlabel(col_label, fontsize=fontsize)

    if save_plot:
        folder = URBAN_EXTRACTION_PATH / 'plots' / 'testing' / 'qualitative' / 'assembled'
        folder.mkdir(exist_ok=True)
        file = folder / f'test_qualitative_results.png'
        plt.savefig(file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def run_quantitative_inference(config_name: str):

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()

    # loading dataset from config (requires inference.json)
    dataset = SpaceNet7Dataset(cfg)

    data = {}
    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            sample = dataset.__getitem__(index)
            img = sample['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).flatten().cpu().numpy()
            y_true = sample['y'].flatten().cpu().numpy()

            group_name = sample['group_name']

            if group_name not in data.keys():
                data[group_name] = {
                    'y_probs': np.array([]),
                    'y_trues': np.array([])
                }

            y_probs = data[group_name]['y_probs']
            y_trues = data[group_name]['y_trues']
            data[group_name] = {
                'y_probs': np.concatenate((y_probs, y_prob), axis=0),
                'y_trues': np.concatenate((y_trues, y_true), axis=0)
            }
        output_file = ROOT_PATH / 'testing' / f'probabilities_{config_name}.npy'
        output_file.parent.mkdir(exist_ok=True)
        np.save(output_file, data)


def plot_precision_recall_curve(config_names: list, group_name: str = None, names: list = None,
                                show_legend: bool = False):

    fig, ax = plt.subplots()
    fontsize = 18
    mpl.rcParams.update({'font.size': fontsize})

    # getting data and if not available produce
    for i, config_name in enumerate(config_names):
        data_file = ROOT_PATH / 'testing' / f'probabilities_{config_name}.npy'
        if not data_file.exists():
            run_quantitative_inference(config_name)
        data = np.load(data_file, allow_pickle=True)
        data = data[()]
        if group_name:
            y_trues = data[group_name]['y_trues']
            y_probs = data[group_name]['y_probs']
        else:
            for i, group_data in enumerate(data.values()):
                if i == 0:
                    y_trues = group_data['y_trues']
                    y_probs = group_data['y_probs']
                else:
                    y_trues = np.concatenate((y_trues, group_data['y_trues']), axis=0)
                    y_probs = np.concatenate((y_probs, group_data['y_probs']), axis=0)
        prec, rec, thresholds = precision_recall_curve(y_trues, y_probs)

        label = config_name if names is None else names[i]
        ax.plot(rec, prec, label=label)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xlabel('Recall', fontsize=fontsize)
        ax.set_ylabel('Precision', fontsize=fontsize)
        ax.set_aspect('equal', adjustable='box')
        ticks = np.linspace(0, 1, 6)
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=fontsize)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=fontsize)
        if show_legend:
            ax.legend()

    prefix = group_name if group_name is not None else 'sn7'
    plot_file = ROOT_PATH / 'plots' / 'precision_recall_curve' / f'{prefix}_{"".join(config_names)}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_threshold_dependency(config_names: list, names: list = None, show_legend: bool = False):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fontsize = 18
    mpl.rcParams.update({'font.size': fontsize})

    # getting data and if not available produce
    for i, config_name in enumerate(config_names):
        data_file = ROOT_PATH / 'testing' / f'probabilities_{config_name}.npy'
        if not data_file.exists():
            run_quantitative_inference(config_name)
        data = np.load(data_file, allow_pickle=True)
        data = data[()]
        for i, group_data in enumerate(data.values()):
            if i == 0:
                y_trues = group_data['y_trues']
                y_probs = group_data['y_probs']
            else:
                y_trues = np.concatenate((y_trues, group_data['y_trues']), axis=0)
                y_probs = np.concatenate((y_probs, group_data['y_probs']), axis=0)

        f1_scores, precisions, recalls = [], [], []
        thresholds = np.linspace(0, 1, 101)
        for thresh in thresholds:
            y_preds = y_probs >= thresh
            tp = np.sum(np.logical_and(y_trues, y_preds))
            fp = np.sum(np.logical_and(y_preds, np.logical_not(y_trues)))
            fn = np.sum(np.logical_and(y_trues, np.logical_not(y_preds)))
            prec = tp / (tp + fp)
            precisions.append(prec)
            rec = tp / (tp + fn)
            recalls.append(rec)
            f1 = 2 * (prec * rec) / (prec + rec)
            f1_scores.append(f1)
        label = config_name if names is None else names[i]

        axs[0].plot(thresholds, f1_scores, label=label)
        axs[1].plot(thresholds, precisions, label=label)
        axs[2].plot(thresholds, recalls, label=label)

    for ax, metric in zip(axs, ['F1 score', 'Precision', 'Recall']):
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xlabel('Threshold', fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize)
        ax.set_aspect('equal', adjustable='box')
        ticks = np.linspace(0, 1, 6)
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=fontsize)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=fontsize)
        if show_legend:
            ax.legend()

    dataset_ax = axs[-1].twinx()
    dataset_ax.set_ylabel('Test', fontsize=fontsize, rotation=270, va='bottom')
    dataset_ax.set_yticks([])

    plot_file = ROOT_PATH / 'plots' / 'f1_curve' / f'sn7_{"".join(config_names)}.png'
    plot_file.parent.mkdir(exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':


    # run_quantitative_inference('fusion')

    # qualitative_testing('sar', 100, save_plots=False)
    # advanced_qualitative_testing('fusion_color', 100, save_plots=False)
    # plot_reference_comparison(40)
    # out_of_distribution_check(60, save_plots=False)
    # out_of_distribution_correlation('optical', 100, save_plot=False)

    # quantitative_testing('fusion', threshold=0.5, save_output=True)
    # qualitative_testing('sar', False)
    # plot_quantitative_testing(['sar', 'optical', 'fusion', 'fusiondual_semisupervised_extended'],
    #                           ['SAR', 'Optical', 'Fusion', 'Fusion-DA'])

    config_names = ['sar', 'optical', 'fusion', 'fusiondual_semisupervised_extended']
    names = ['SAR', 'Optical', 'Fusion', 'Fusion-DA']
    # plot_activation_comparison(config_names, save_plots=True)
    # for config_name in config_names:
    #     show_quantitative_testing(config_name)
    aoi_ids = [
        'L15-0506E-1204N_2027_3374_13',
        'L15-0595E-1278N_2383_3079_13',
        'L15-1172E-1306N_4688_2967_13',
        'L15-0632E-0892N_2528_4620_13',
        'L15-1209E-1113N_4838_3737_13',
        'L15-1015E-1062N_4061_3941_13',
        'L15-1204E-1202N_4816_3380_13',
        'L15-0977E-1187N_3911_3441_13',
        'L15-1672E-1207N_6691_3363_13',


    ]
    # plot_activation_comparison_assembled(config_names, names, aoi_ids, save_plot=True)
    plot_activation_comparison(config_names, save_plots=True)
    # quantitative_testing('sar_confidence', True)
    # plot_precision_recall_curve(['optical', 'sar', 'fusion', 'fusiondual_semisupervised'], 'SA')
    # plot_threshold_dependency(['optical', 'sar', 'fusion', 'fusiondual_semisupervised'])

