import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utils.geotiff import *
import numpy as np
from pathlib import Path


def plot_optical(ax, file: Path, vis: str = 'true_color', scale_factor: float = 0.3):
    img, _, _ = read_tif(file)
    band_indices = [2, 1, 0] if vis == 'true_color' else [6, 2, 1]
    bands = img[:, :, band_indices] / scale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)


def plot_sar(ax, file: Path, vis: str = 'VV'):
    img, _, _ = read_tif(file)
    band_index = 0 if vis == 'VV' else 1
    bands = img[:, :, band_index]
    bands = bands.clip(0, 1)
    ax.imshow(bands)


def plot_buildings(ax, file: Path):
    img, _, _ = read_tif(file)
    img = img > 0
    cmap = colors.ListedColormap(['white', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(img, cmap=cmap, norm=norm)