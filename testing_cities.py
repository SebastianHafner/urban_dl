from pathlib import Path
import torch
from networks.network_loader import load_network
from experiment_manager.config import config
from utils.datasets import TilesInferenceDataset
from utils.geotiff import *
from tqdm import tqdm
import numpy as np
from utils.metrics import *

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def run_inference(config_name: str, site: str):
    print(f'running inference for {site}...')

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


def run_quantitative_evaluation(config_name: str, site: str, threshold: float = None):
    print(f'running inference for {site}...')

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{config_name}_{cfg.INFERENCE.CHECKPOINT}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = TilesInferenceDataset(cfg, site)

    y_preds, y_trues = None, None

    thresh = threshold if threshold else cfg.INFERENCE.THRESHOLDS.VALIDATION

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            img = patch['x'].to(device)
            logits = net(img.unsqueeze(0))
            prob = torch.sigmoid(logits).squeeze()
            pred = prob > thresh
            center_pred = pred[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]
            center_pred = center_pred.flatten().float()

            assert (patch['is_labeled'])
            label = patch['y'].to(device)
            label = label.flatten().float()

            if y_preds is not None:
                y_preds = torch.cat((y_preds, center_pred), dim=0)
                y_trues = torch.cat((y_trues, label), dim=0)
            else:
                y_preds = center_pred
                y_trues = label

        prec = precision(y_trues, y_preds, dim=0)
        rec = recall(y_trues, y_preds, dim=0)
        f1 = f1_score(y_trues, y_preds, dim=0)
        print(f'{site}: f1 score {f1:.3f} - precision {prec:.3f} - recall {rec:.3f}')




if __name__ == '__main__':
    config_name = 'igarss_sar'
    cities = ['nanning', 'santiagodechile', 'beirut', 'lagos', 'newdehli']
    cities_igarss = ['stockholm', 'kampala', 'daressalam', 'sidney']
    for city in cities_igarss:
        # run_inference(config_name, city)
        run_quantitative_evaluation(config_name, city, threshold=0.5)
