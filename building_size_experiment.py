from utils.visualization import *
from pathlib import Path
import numpy as np
import utm
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


DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks')
SN7_PATH = Path('/storage/shafner/urban_extraction/spacenet7')

GROUPS = [(1, 'NA_AU', '#63cd93'), (2, 'SA', '#f0828f'), (3, 'EU', '#6faec9'), (4, 'SSA', '#5f4ad9'),
          (5, 'NAF_ME', '#8dee47'), (6, 'AS', '#d9b657'), ('total', 'Total', '#ffffff')]


def polygonareanp(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def centroidnp(arr: np.ndarray) -> np.ndarray:
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def centroid(feature: dict) -> tuple:
    coords = feature['geometry']['coordinates']
    coords = np.array(coords)[0, ]
    c = centroidnp(coords)
    lon, lat = c
    return lat, lon


def value_at_coords(img: np.ndarray, transform, coords) -> float:
    x_coord, y_coord = coords

    y_pixels, x_pixel = img.shape
    x_pixel_spacing = transform[0]
    x_min = transform[2]
    x_index = int((x_coord - x_min) // x_pixel_spacing)


    y_pixel_spacing = transform[4]
    y_max = transform[5]
    y_index = int((y_coord - y_max) // y_pixel_spacing)

    value = img[x_index, y_index]
    return value


def load_building_footprints(aoi_id: str, year: int, month: int):
    file_name = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
    label_file = SN7_PATH / 'train' / aoi_id / 'labels' / file_name
    label = load_json(label_file)
    features = label['features']
    return features


def run_building_size_experiment(config_name: str, checkpoint: int):
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
    thresh = cfg.THRESH

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)
        x = sample['x'].to(device)
        y_true = sample['y'].to(device)
        transform, crs = sample['transform'], sample['crs']

        y_true = y_true.detach().cpu().flatten().numpy()[0, ]
        building_footprints = load_building_footprints(sample['aoi_id'], sample['year'], sample['month'])
        for footprint in building_footprints:
            building_centroid = centroid(footprint)
            # TODO: use numpy array instead
            easting, northing, zone_number, zone_letter = utm.from_latlon(*building_centroid)



            prob = value_at_coords(y_true, transform, (easting, northing))



        # with torch.no_grad():
        #
        #     logits = net(x.unsqueeze(0))
        #     y_pred = torch.sigmoid(logits) > thresh
        #
        #     y_true = y_true.detach().cpu().flatten().numpy()
        #     y_pred = y_pred.detach().cpu().flatten().numpy()



if __name__ == '__main__':
    config_name = 'baseline_sar'
    checkpoint = 100
    run_building_size_experiment(config_name, checkpoint)


