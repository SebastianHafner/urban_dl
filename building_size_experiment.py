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
    # TODO: solve for polygons that have list of coordinates (i.e. inner polygon)
    coords = np.array(coords[0])
    c = centroidnp(coords)
    lon, lat = c
    return lat, lon


def unpack_transform(arr, transform):

    # TODO: look for a nicer solution here
    if len(arr.shape) == 2:
        y_pixels, x_pixels = arr.shape
    else:
        y_pixels, x_pixels, *_ = arr.shape

    x_pixel_spacing = transform[0]
    x_min = transform[2]
    x_max = x_min + x_pixels * x_pixel_spacing

    y_pixel_spacing = transform[4]
    y_max = transform[5]
    y_min = y_pixels * y_pixel_spacing

    return x_min, x_max, x_pixel_spacing, y_min, y_max, y_pixel_spacing


def is_out_of_bounds(img: np.ndarray, transform, coords) -> bool:
    x_coord, y_coord = coords
    x_min, x_max, _, y_min, y_max, _ = unpack_transform(img, transform)
    if x_coord < x_min or x_coord > x_max:
        return True
    if y_coord < y_min or y_coord > y_max:
        return True
    return False


def is_valid_footprint(footprint):
    if footprint['geometry'] is None:
        return False
    elif footprint['geometry']['type'] != 'Polygon':
        return False
    else:
        return True


def value_at_coords(img: np.ndarray, transform, coords) -> float:
    x_coord, y_coord = coords
    x_min, _, x_pixel_spacing, _, y_max, y_pixel_spacing = unpack_transform(img, transform)
    x_index = int((x_coord - x_min) // x_pixel_spacing)
    y_index = int((y_coord - y_max) // y_pixel_spacing)
    value = img[y_index, x_index, ]
    return value


def load_building_footprints(aoi_id: str, year: int, month: int, wgs84: bool = True):
    file_name = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
    label_folder = 'labels' if wgs84 else 'labels_match'
    label_file = SN7_PATH / 'train' / aoi_id / label_folder / file_name
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

    output_data = []

    for index in range(len(dataset)):
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        print(f'processing {aoi_id}')

        x = sample['x'].to(device)
        y_true = sample['y'].to(device)
        transform, crs = sample['transform'], sample['crs']

        with torch.no_grad():
            logits = net(x.unsqueeze(0))
            y_prob = torch.sigmoid(logits)
            y_true = torch.squeeze(y_true)
            y_prob = torch.squeeze(y_prob)

            y_true = y_true.detach().cpu().numpy()
            y_prob = y_prob.detach().cpu().numpy()

        footprints = load_building_footprints(sample['aoi_id'], sample['year'], sample['month'], wgs84=False)
        areas = [footprint['properties']['area'] for footprint in footprints]
        footprints = load_building_footprints(sample['aoi_id'], sample['year'], sample['month'])

        for i, (footprint, area) in enumerate(zip(footprints, areas)):
            if is_valid_footprint(footprint):
                building_centroid = centroid(footprint)
                easting, northing, zone_number, zone_letter = utm.from_latlon(*building_centroid)

                # check for out of bounds
                if not is_out_of_bounds(y_prob, transform, (easting, northing)):
                    prob = value_at_coords(y_prob, transform, (easting, northing))
                    output_data.append((float(area), float(prob)))

    output_file = DATASET_PATH.parent / 'building_size_experiment' / f'data_{config_name}.json'
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def plot_building_size_experiment(config_name: str):
    file = DATASET_PATH.parent / 'building_size_experiment' / f'data_{config_name}.json'
    data = load_json(file)
    areas = [d[0] for d in data]
    prob = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(areas, prob)
    plt.show()







if __name__ == '__main__':
    config_name = 'baseline_sar'
    checkpoint = 100
    # run_building_size_experiment(config_name, checkpoint)
    plot_building_size_experiment(config_name)
