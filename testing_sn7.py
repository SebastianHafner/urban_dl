from utils.visualization import *
from pathlib import Path
import numpy as np
import json
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

mpl.rcParams.update({'font.size': 20})

GROUPS = [(1, 'NA_AU', '#63cd93'), (2, 'SA', '#f0828f'), (3, 'EU', '#6faec9'), (4, 'SSA', '#5f4ad9'),
          (5, 'NAF_ME', '#8dee47'), (6, 'AS', '#d9b657'), ('total', 'Total', '#ffffff')]
GROUP_NAMES = ['NA_AU', 'SA', 'EU', 'SSA', 'NAF_ME', 'AS']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


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
            test_site = dataset.__getitem__(index)
            aoi_id = test_site['aoi_id']
            if aoi_id == 'L15-0434E-1218N_1736_3318_13':
                continue
            img = test_site['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).flatten().cpu().numpy()
            y_true = test_site['y'].flatten().cpu().numpy()

            group_name = test_site['group_name']
            if group_name not in data.keys():
                data[group_name] = []

            f1 = f1_score_from_prob(y_prob, y_true, 0.5)
            p = precsision_from_prob(y_prob, y_true, 0.5)
            r = recalll_from_prob(y_prob, y_true, 0.5)

            site_data = {
                'aoi_id': test_site['aoi_id'],
                'y_prob': y_prob,
                'y_true': y_true,
                'f1_score': f1,
                'precision': p,
                'recall': r,
            }

            data[group_name].append(site_data)

        output_file = ROOT_PATH / 'testing' / f'probabilities_{config_name}.npy'
        output_file.parent.mkdir(exist_ok=True)
        np.save(output_file, data)


def get_quantitative_data(config_name: str, allow_run: bool = True):
    data_file = ROOT_PATH / 'testing' / f'probabilities_{config_name}.npy'
    if not data_file.exists():
        if allow_run:
            run_quantitative_inference(config_name)
        else:
            raise Exception('No data and not allowed to run quantitative inference!')
    run_quantitative_inference(config_name)
    data = np.load(data_file, allow_pickle=True)
    data = dict(data[()])
    return data



def plot_boxplots(config_names: list, names: list = None):


    metrics = ['f1_score', 'precision', 'recall']
    metric_names = ['F1 score', 'Precision', 'Recall']
    box_width = 0.2
    wisk_width = 0.1
    line_width = 2
    point_size = 40

    for metric, metric_name in zip(metrics, metric_names):
        fig, ax = plt.subplots(figsize=(10, 5))

        def custom_boxplot(x_pos: float, values: list, color: str):
            min_ = np.min(values)
            max_ = np.max(values)
            median = np.median(values)

            line_kwargs = {'c': color, 'lw': line_width}
            point_kwargs = {'c': color, 's': point_size}

            # vertical line
            ax.plot(2 * [x_pos], [min_, max_], **line_kwargs)

            # whiskers
            x_positions = [x_pos - wisk_width / 2, x_pos + wisk_width / 2]
            ax.plot(x_positions, [min_, min_], **line_kwargs)
            ax.plot(x_positions, [max_, max_], **line_kwargs)

            # median
            ax.scatter([x_pos], [median], **point_kwargs)

            pass

        for i, config_name in enumerate(config_names):

            # x data
            # positions = np.arange(len(GROUP_NAMES)) + (i * box_width)

            # y data
            data = get_quantitative_data(config_name)
            # boxplot_data = [[site[metric] for site in data[group_name]] for group_name in GROUP_NAMES]

            for j, group_name in enumerate(GROUP_NAMES):
                values = [site[metric] for site in data[group_name]]
                x_pos = j + (i * box_width)
                custom_boxplot(x_pos, values, color=COLORS[i])

            # ax.boxplot(boxplot_data, positions=positions, widths=box_width, zorder=3, whis=[0, 100], sym='', notch=False,
            #            showbox=False, conf_intervals=conf_intervals)

        ax.set_ylim((0, 1))
        ax.set_ylabel(metric_name)

        x_ticks = np.arange(len(GROUP_NAMES)) + (len(config_names) - 1) * box_width / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(GROUP_NAMES)
        # plt.grid(b=True, which='major', axis='y', zorder=0)

        if metric == 'f1_score':
            handles = [Line2D([0], [0], color=COLORS[i], lw=line_width) for i in range(len(config_names))]
            ax.legend(handles, names, loc='upper center', ncol=4, frameon=False, handletextpad=0.8,
                      columnspacing=1, handlelength=1)
        plt.show()
        plt.close(fig)


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



def plot_barplots(config_names: list, names: list):

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



def plot_activation_histograms(config_names: str):
    plot_size = 2
    n_cols = len(config_names)
    n_rows = len(GROUP_NAMES)
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(plot_size * n_cols, plot_size * n_rows))
    for j, config_name in enumerate(config_names):
        data_file = ROOT_PATH / 'testing' / f'probabilities_{config_name}.npy'
        if not data_file.exists():
            run_quantitative_inference(config_name)
        data = np.load(data_file, allow_pickle=True)
        data = data[()]

        for i, group_name in enumerate(GROUP_NAMES):
            group_data = data[group_name]
            probs = group_data['y_probs']

            bin_edges = np.linspace(0, 1, 21)
            ax = axs[i, j]
            ax.hist(probs, weights=np.zeros_like(probs) + 1. / probs.size, bins=bin_edges, range=(0, 1))
            ax.set_xlim((0, 1))
            ax.set_xticks(np.linspace(0, 1, 5))
            # ax.set_yscale('log')
            # ax.axvline(x=thresh, color='k')
            ax.set_ylim((0, 0.1))
            # ax.set_aspect('equal', adjustable='box')
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

    config_name = 'fusiondual_semisupervised_extended'
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
    # plot_activation_comparison(config_names, save_plots=True)
    # quantitative_testing('sar_confidence', True)
    # plot_precision_recall_curve(['optical', 'sar', 'fusion', 'fusiondual_semisupervised'], 'SA')
    # plot_threshold_dependency(['optical', 'sar', 'fusion', 'fusiondual_semisupervised'])
    # plot_activation_histograms(config_names)
    plot_boxplots(config_names, names)
