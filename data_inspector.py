
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from utils.geotiff import *
from utils.visualization import *
import json
import pandas as pd

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
DATASET_PATH = Path('C:/Users/shafner/urban_extraction/data/dummy_data')
METADATA_FILE = Path('C:/Users/shafner/urban_extraction/data/spacenet7/sn7_metadata.csv')


def show_patch_triplet(site: str, patch_id):
    s1_file = DATASET_PATH / site / 'sentinel1' / f'sentinel1_{site}_patch{patch_id}.tif'
    s2_file = DATASET_PATH / site / 'sentinel2' / f'sentinel2_{site}_patch{patch_id}.tif'
    buildings_file = DATASET_PATH / site / 'buildings' / f'buildings_{site}_patch{patch_id}.tif'

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    plot_optical(axs[0], s2_file)
    plot_sar(axs[1], s1_file)
    plot_buildings(axs[2], buildings_file)
    plt.title(f'{site} {patch_id}')
    plt.show()


def show_satellite_data_sn7(aoi_id: str, title: str):
    s1_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
    s2_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    plot_optical(axs[0], s2_file)
    plot_optical(axs[1], s2_file, vis='false_color')
    plot_sar(axs[2], s1_file)
    plt.title(title)
    plt.show()


def show_patch_triplet_sn7(aoi_id: str, title: str):
    s1_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
    s2_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
    buildings_file = DATASET_PATH / 'sn7' / 'buildings' / f'buildings_{aoi_id}.tif'

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    plot_optical(axs[0], s2_file)
    plot_sar(axs[1], s1_file)
    plot_buildings(axs[2], buildings_file)
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    # corrupt patches
    # losangeles_patch_ids = [159, 158, 145]
    # dallas_patch_ids = [100, 115, 133, 16, 132, 61, 5, 179, 163, 58, 38]
    # toronto_patch_ids = [193, 269, 215, 187, 246, 286, 76, 50, 228, 205, 7, 180, 184, 48, 289, 14, 84, 329, 19, 102,
    #                      128, 293, 49, 194, 75, 186, 125, 156, 44, 296, 8, 158, 218, 225, 291, 27, 29]
    # montreal_patch_ids = [104, 84, 119, 37, 39, 13, 73, 8, 56, 48, 78, 99, 120, 4, 46, 116, 47, 103, 42, 75, 82, 41,
    #                       100, 71, 106, 108, 95, 90, 29, 57, 86, 5, 27, 12, 51, 66, 32, 2, 91, 76, 30, 59, 92, 60, 54,
    #                       28, 89, 31, 55, 112, 94, 21, 122, 17, 80, 45, 10]
    #
    # city = 'montreal'
    # patch_ids = montreal_patch_ids
    # for patch_id in patch_ids:
    #     show_patch_triplet(city, patch_id)

    metadata = pd.read_csv(METADATA_FILE)
    for index, row in metadata.iterrows():
        print(index)
        aoi_id = str(row['aoi_id'])
        # show_satellite_data_sn7(aoi_id, f'{index} {aoi_id}')
        show_patch_triplet_sn7(aoi_id, f'{index} {aoi_id}')