
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.geotiff import *


def visualize_triplet(path: Path, city: str, patch_id: int):

    S2_BAND_INDICES = [2, 1, 0]

    s1_file = path / city / 'sentinel1' / f'sentinel1_{city}_patch{patch_id}.tif'
    s2_file = path / city / 'sentinel2' / f'sentinel2_{city}_patch{patch_id}.tif'
    buildings_file = path / city / 'buildings' / f'buildings_{city}_patch{patch_id}.tif'

    s1_data, _, _ = read_tif(s1_file)
    s2_data, _, _ = read_tif(s2_file)
    buildings_data, _, _ = read_tif(buildings_file)

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(s2_data[:, :, S2_BAND_INDICES])
    axs[1].imshow(s1_data[:, :, 0])
    axs[2].imshow(buildings_data[:, :, 0])

    for ax in axs:
        ax.set_axis_off()

    plt.show()


if __name__ == '__main__':

    DATASET_PATH = Path('/storage/shafner/urban_extraction/')
    CITY = 'miami'

    n = 5
    max_patch_id = 100
    # TODO: get maximum patch id for city
    patch_ids = np.random.randint(0, max_patch_id, n)
    for patch_id in range(n):
        visualize_triplet(DATASET_PATH, CITY, patch_id)
