import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utils.geotiff import *
import numpy as np
from pathlib import Path


def plot_optical(ax, file: Path, vis: str = 'true_color', scale_factor: float = 0.4,
                 show_title: bool = False):
    img, _, _ = read_tif(file)
    band_indices = [2, 1, 0] if vis == 'true_color' else [6, 2, 1]
    bands = img[:, :, band_indices] / scale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title(f'optical ({vis})')


def plot_sar(ax, file: Path, vis: str = 'VV', show_title: bool = False):
    img, _, _ = read_tif(file)
    band_index = 0 if vis == 'VV' else 1
    bands = img[:, :, band_index]
    bands = bands.clip(0, 1)
    ax.imshow(bands, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title(f'sar ({vis})')


def plot_buildings(ax, file: Path, show_title: bool = False):
    img, _, _ = read_tif(file)
    img = img > 0
    img = img if len(img.shape) == 2 else img[:, :, 0]
    cmap = colors.ListedColormap(['white', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(img, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('ground truth')


def plot_activation(ax, activation: np.ndarray, show_title: bool = False):
    ax.imshow(activation, cmap='jet', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('activation')


def plot_prediction(ax, prediction: np.ndarray, show_title: bool = False):
    cmap = colors.ListedColormap(['white', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(prediction, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('prediction')
