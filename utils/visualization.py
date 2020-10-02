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


def plot_stable_buildings(ax, file_all: Path, file_stable: Path, show_title: bool = False):
    img_all, _, _ = read_tif(file_all)
    img_all = img_all > 0
    img_all = img_all if len(img_all.shape) == 2 else img_all[:, :, 0]

    img_stable, _, _ = read_tif(file_stable)
    img_stable = img_stable > 0
    img_stable = img_stable if len(img_stable.shape) == 2 else img_stable[:, :, 0]

    img_instable = np.logical_and(img_all, np.logical_not(img_stable)) * 2

    cmap = colors.ListedColormap(['white', 'red', 'blue'])
    boundaries = [0, 0.5, 1, 1.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(img_all + img_instable, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('ground truth')


def plot_probability(ax, probability: np.ndarray, show_title: bool = False):
    ax.imshow(probability, cmap='jet', vmin=0, vmax=1)
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


def plot_probability_histogram(ax, probability: np.ndarray, show_title: bool = False):

    bin_edges = np.linspace(0, 1, 21)
    values = probability.flatten()
    ax.hist(values, bins=bin_edges, range=(0, 1))
    ax.set_xlim((0, 1))
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yscale('log')

    if show_title:
        ax.set_title('probability histogram')


if __name__ == '__main__':
    arr = np.array([[0, 0.01, 0.1, 0.89, 0.9, 1, 1, 1]]).flatten()
    # hist, bin_edges = np.histogram(arr, bins=10, range=(0, 1))
