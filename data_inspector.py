
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from utils.geotiff import *
from utils.visualization import *
import json
import pandas as pd
from tqdm import tqdm

ROOT_PATH = Path('/storage/shafner/urban_extraction')
DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction_dataset')
METADATA_FILE = Path('C:/Users/shafner/urban_extraction/data/spacenet7/sn7_metadata_v3.csv')

GROUPS = {1: 'NA_AU', 2: 'SA', 3: 'EU', 4: 'SSA', 5: 'NAF_ME', 6: 'AS'}


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

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plot_optical(axs[0], s2_file)
    plot_optical(axs[1], s2_file, vis='false_color')
    # plot_sar(axs[2], s1_file)
    plt.suptitle(title, size=20)
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


def show_patches(city: str, n: int):
    s1_path = DATASET_PATH / city / 'sentinel1'
    patches = [f.stem.split('_')[-1] for f in s1_path.glob('**/*')]

    max_samples = n if n <= len(patches) else len(patches)
    indices = np.arange(0, max_samples)

    for index in tqdm(indices):
        patch_id = patches[index]

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))

        s2_file = DATASET_PATH / city / 'sentinel2' / f'sentinel2_{city}_{patch_id}.tif'
        plot_optical(axs[0], s2_file)

        s1_file = DATASET_PATH / city / 'sentinel1' / f'sentinel1_{city}_{patch_id}.tif'
        plot_sar(axs[1], s1_file)

        buildings_file = DATASET_PATH / city / 'buildings' / f'buildings_{city}_{patch_id}.tif'
        if buildings_file.exists():
            plot_buildings(axs[2], buildings_file)
        plt.suptitle(f'{city} {patch_id}')
        plt.show()
        plt.close()


def patches2png(site: str, sensor: str, band_indices: list, rescale_factor: float = 1, patch_size: int = 256):
    # config inference directory
    save_path = ROOT_PATH / 'plots' / 'inspection'
    save_path.mkdir(exist_ok=True)

    folder = ROOT_PATH / 'urban_extraction_dataset' / site / sensor
    files = [f for f in folder.glob('**/*')]

    files_per_side = int(np.ceil(np.sqrt(len(files))))
    arr = np.zeros((files_per_side * patch_size, files_per_side * patch_size, 3))
    for index, f in enumerate(tqdm(files)):
        patch, _, _ = read_tif(f)
        m, n, _ = patch.shape
        i = (index // files_per_side) * patch_size
        j = (index % files_per_side) * patch_size
        if len(band_indices) == 3:
            arr[i:i + m, j:j + n, ] = patch[:, :, band_indices]
        else:
            for b in range(3):
                arr[i:i + m, j:j + n, b:b + 1] = patch[:, :, band_indices]

    plt.imshow(np.clip(arr / rescale_factor, 0, 1))
    plt.axis('off')
    save_file = save_path / f'{site}_{sensor}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    show_patches('kigali', 20)

    # metadata = pd.read_csv(METADATA_FILE)
    # for index, row in metadata.iterrows():
    #     if index >= 47:
    #         print(index)
    #         aoi_id = str(row['aoi_id'])
    #         group_nr = int(row['group'])
    #         group = GROUPS[group_nr]
    #         year = int(row['year'])
    #         month = int(row['month'])
    #         country = str(row['country'])
    #
    #         title = f'{index} {aoi_id} {year}-{month:02d} ({country}, {group})'
    #         show_satellite_data_sn7(aoi_id, title)
    all_sites = ['denver', 'saltlakecity', 'phoenix', 'lasvegas', 'toronto', 'columbus', 'winnipeg', 'dallas',
                 'minneapolis', 'atlanta', 'miami', 'montreal', 'quebec', 'albuquerque', 'losangeles', 'kansascity',
                 'charlston', 'seattle', 'elpaso', 'sandiego', 'santafe', 'stgeorge', 'tucson', 'houston',
                 'sanfrancisco',
                 'vancouver', 'newyork', 'calgary', 'kampala', 'stockholm', 'beijing', 'jakarta',
                 'kigali', 'lagos', 'mexicocity', 'milano', 'mumbai', 'riodejanairo', 'sidney'
                 ]
    for site in all_sites:
        for sensor, indices, factor in zip(['sentinel1', 'sentinel2'], [[0], [2, 1, 0]], [1, 0.4]):
            patches2png(site, sensor, indices, factor)