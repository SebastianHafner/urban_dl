from pathlib import Path
import torch
from networks.network_loader import load_network
from experiment_manager.config import config
from utils.datasets import TilesInferenceDataset
from utils.geotiff import *
from tqdm import tqdm
import numpy as np
from utils.metrics import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import precision_recall_curve

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def run_inference(config_name: str, site: str):
    print(f'running inference for {site} with {config_name}...')

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = TilesInferenceDataset(cfg, site)

    # config inference directory
    save_path = ROOT_PATH / 'inference' / config_name
    save_path.mkdir(exist_ok=True)

    prob_output = dataset.get_arr()
    transform, crs = dataset.get_geo()

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            img = patch['x'].to(device)
            logits = net(img.unsqueeze(0))
            prob = torch.sigmoid(logits) * 100
            prob = prob.squeeze().cpu().numpy().astype('uint8')
            prob = np.clip(prob, 0, 100)
            center_prob = prob[dataset.patch_size:dataset.patch_size*2, dataset.patch_size:dataset.patch_size*2]

            i_start = patch['i']
            i_end = i_start + dataset.patch_size
            j_start = patch['j']
            j_end = j_start + dataset.patch_size
            prob_output[i_start:i_end, j_start:j_end, 0] = center_prob

    output_file = save_path / f'prob_{site}_{config_name}.tif'
    write_tif(output_file, prob_output, transform, crs)


def run_quantitative_evaluation(config_name: str, site: str, threshold: float = None, save_output: bool = False):
    print(f'running quantitative evaluation for {site} with {config_name}...')

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = TilesInferenceDataset(cfg, site)

    y_probs, y_trues = None, None

    thresh = threshold if threshold else cfg.INFERENCE.THRESHOLDS.VALIDATION

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            img = patch['x'].to(device)
            logits = net(img.unsqueeze(0))
            prob = torch.sigmoid(logits).squeeze()

            center_prob = prob[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]
            center_prob = center_prob.flatten().float().cpu()

            assert (patch['is_labeled'])
            label = patch['y'].flatten().float().cpu()

            if y_probs is not None:
                y_probs = torch.cat((y_probs, center_prob), dim=0)
                y_trues = torch.cat((y_trues, label), dim=0)
            else:
                y_probs = center_prob
                y_trues = label

        if save_output:
            y_probs = y_probs.numpy()
            y_trues = y_trues.numpy()
            output_data = np.stack((y_trues, y_probs))
            output_path = ROOT_PATH / 'quantitative_evaluation' / config_name
            output_path.mkdir(exist_ok=True)
            output_file = output_path / f'{site}_{config_name}.npy'
            np.save(output_file, output_data)
        else:
            y_preds = (y_probs > thresh).float()
            prec = precision(y_trues, y_preds, dim=0)
            rec = recall(y_trues, y_preds, dim=0)
            f1 = f1_score(y_trues, y_preds, dim=0)
            print(f'{site}: f1 score {f1:.3f} - precision {prec:.3f} - recall {rec:.3f}')


def plot_precision_recall_curve(site: str, config_names: list, names: list = None, show_legend: bool = False):

    fig, ax = plt.subplots()
    fontsize = 14
    mpl.rcParams.update({'font.size': fontsize})

    # getting data and if not available produce
    for i, config_name in enumerate(config_names):
        data_file_config = ROOT_PATH / 'quantitative_evaluation' / config_name / f'{site}_{config_name}.npy'
        if not data_file_config.exists():
            run_quantitative_evaluation(config_name, site, save_output=True)
        data_config = np.load(data_file_config)
        prec, rec, thresholds = precision_recall_curve(data_config[0, ], data_config[1, ])

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
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    config_name = 'igarss_optical'
    cities = ['nanning', 'santiagodechile', 'beirut', 'lagos', 'newdehli']
    cities_igarss = ['stockholm', 'kampala', 'daressalam', 'sidney']
    for i, city in enumerate(cities_igarss):
        legend = True if i == 0 else False
        # run_inference(config_name, city)
        # run_quantitative_evaluation(config_name, city, threshold=0.5, save_output=True)

        plot_precision_recall_curve(city, ['igarss_sar', 'igarss_optical', 'igarss_fusion', 'ghsl'],
                                    names=['SAR', 'Optical', 'Fusion', 'GHSL'], show_legend=legend)
