from pathlib import Path
from networks.network_loader import load_network
from utils.datasets import UrbanExtractionDataset
from experiment_manager.config import config
from utils.metrics import *
from tqdm import tqdm
from torch.utils import data as torch_data
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from utils.visualization import *
import matplotlib as mpl

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')
ROOT_PATH = Path('/storage/shafner/urban_extraction')


def quantitative_validation(config_name: str, checkpoint: int, save_output: bool = False):

    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    thresh = cfg.THRESHOLDS.TRAIN

    output_data = {'training': {}, 'validation': {}}
    # training can be added
    for run_type in ['validation']:
        sites = cfg.DATASETS.SITES.TRAINING if run_type == 'training' else cfg.DATASETS.SITES.VALIDATION
        for site in sites:
            print(f'Quantitative assessment {site} ({run_type})')
            dataset = UrbanExtractionDataset(cfg=cfg, dataset=site, no_augmentations=True)

            dataloader_kwargs = {
                'batch_size': cfg.TRAINER.BATCH_SIZE,
                'num_workers': cfg.DATALOADER.NUM_WORKER,
                'shuffle': False,
                'drop_last': False,
                'pin_memory': True,
            }
            dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

            y_true_set, y_pred_set = np.array([]), np.array([])
            for i, batch in enumerate(tqdm(dataloader)):
                with torch.no_grad():
                    x = batch['x'].to(device)
                    y_true = batch['y'].to(device)
                    logits = net(x)
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


def random_selection(config_name: str, checkpoint: int, site: str, n: int):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading dataset
    dataset = UrbanExtractionDataset(cfg, site, no_augmentations=True)

    # loading network
    net = load_network(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    thresh = cfg.THRESHOLDS.TRAIN

    item_indices = list(np.random.randint(0, len(dataset), size=n))

    for index in item_indices:
        sample = dataset.__getitem__(index)
        patch_id = sample['patch_id']

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

        optical_file = DATASET_PATH / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        plot_optical(axs[0, 0], optical_file, vis='false_color', show_title=True)

        sar_file = DATASET_PATH / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        plot_sar(axs[0, 1], sar_file, show_title=True)

        label = cfg.DATALOADER.LABEL
        label_file = DATASET_PATH / site / label / f'{label}_{site}_{patch_id}.tif'
        plot_buildings(axs[0, 2], label_file, show_title=True)

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


        plt.show()


def run_quantitative_inference(config_name: str, run_type: str):
    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net = load_network(cfg, Path(cfg.OUTPUT_BASE_DIR) / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()

    # loading dataset from config (requires inference.json)
    dataset = UrbanExtractionDataset(cfg, dataset=run_type, no_augmentations=True, include_unlabeled=False)

    y_probs = y_trues = None

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            sample = dataset.__getitem__(index)
            img = sample['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).flatten().cpu().numpy()
            y_true = sample['y'].flatten().cpu().numpy()

            if y_probs is None or y_trues is None:
                y_probs, y_trues = np.array([]), np.array([])

            y_probs = np.concatenate((y_probs, y_prob), axis=0)
            y_trues = np.concatenate((y_trues, y_true), axis=0)

        output_file = ROOT_PATH / 'validation' / f'probabilities_{run_type}_{config_name}.npy'
        output_file.parent.mkdir(exist_ok=True)
        output_data = np.stack((y_trues, y_probs))
        np.save(output_file, output_data)


# TODO: this function can be the same shortened by combining it with the one for the test set
def plot_threshold_dependency(config_names: list, run_type: str, names: list = None, show_legend: bool = False):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fontsize = 18
    mpl.rcParams.update({'font.size': fontsize})

    # getting data and if not available produce
    for i, config_name in enumerate(config_names):
        data_file = ROOT_PATH / 'validation' / f'probabilities_{run_type}_{config_name}.npy'
        if not data_file.exists():
            run_quantitative_inference(config_name, run_type)
        data = np.load(data_file, allow_pickle=True)
        y_trues, y_probs = data[0, ], data[1, ]

        f1_scores, precisions, recalls = [], [], []
        thresholds = np.linspace(0, 1, 101)
        print(config_name)
        for thresh in tqdm(thresholds):
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
    dataset_ax.set_ylabel(run_type.capitalize(), fontsize=fontsize, rotation=270, va='bottom')
    dataset_ax.set_yticks([])
    plot_file = ROOT_PATH / 'plots' / 'f1_curve' / f'{run_type}_{"".join(config_names)}.png'
    plot_file.parent.mkdir(exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    config_name = 'optical'

    config_names = ['sar', 'optical', 'fusion', 'fusiondual_semisupervised_extended']
    names = ['SAR', 'Optical', 'Fusion', 'Fusion-DA']
    plot_threshold_dependency(config_names, 'training', names)
