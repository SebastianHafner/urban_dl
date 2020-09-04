
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from utils.geotiff import *
import json


def visualize_triplet(path: Path, sample: dict):

    S2_BAND_INDICES = [2, 1, 0]

    city = sample['city']
    patch_id = sample['patch_id']









    # sentinel-2
    s2_file = path / city / 'sentinel2' / f'sentinel2_{city}_{patch_id}.tif'
    s2_data, _, _ = read_tif(s2_file)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    true_color_img = s2_data[:, :, S2_BAND_INDICES] / 0.3
    axs[0].imshow(true_color_img)

    # sentinel-1
    s1_file = path / city / 'sentinel1' / f'sentinel1_{city}_{patch_id}.tif'
    s1_data, _, _ = read_tif(s1_file)
    vv_img = s1_data[:, :, 0]
    axs[1].imshow(vv_img, cmap='gray')

    # building label
    buildings_file = path / city / 'buildings' / f'buildings_{city}_{patch_id}.tif'
    buildings_data, _, _ = read_tif(buildings_file)
    buildings_img = buildings_data > 0
    cmap = colors.ListedColormap(['white', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    axs[2].imshow(buildings_img, cmap=cmap, norm=norm)

    for ax in axs:
        ax.set_axis_off()

    plt.show()


if __name__ == '__main__':

    DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
    CITY = 'houston'

    n = 10
    samples_file = DATASET_PATH / CITY / 'samples.json'
    with open(str(samples_file)) as f:
        metadata = json.load(f)
        samples = metadata['samples']

    # TODO: get maximum patch id for city
    sample_selection = list(np.random.choice(samples, size=n))
    for sample in sample_selection:
        visualize_triplet(DATASET_PATH, sample)
