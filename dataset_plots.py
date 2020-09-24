import matplotlib.pyplot as plt
from pathlib import Path
from utils.dataloader import UrbanExtractionDataset, SpaceNet7Dataset
from experiment_manager.config import config
from utils.geotiff import *
import torch
import numpy as np
from tqdm import tqdm


DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def plot_number_of_pixels(config_name: str = 'base_v2'):
    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    training_dataset = UrbanExtractionDataset(cfg, 'training', no_augmentations=True)
    validation_dataset = UrbanExtractionDataset(cfg, 'validation', no_augmentations=True)
    test_dataset = SpaceNet7Dataset(cfg)

    datasets = [test_dataset, validation_dataset, training_dataset]
    labels = ['test', 'validation', 'train']
    # container = {'builtup': [], 'background': []}
    # for dataset, label in zip(datasets, labels):
    #     print(f'processing {label}')
    #     builtup_count = 0
    #     for index in tqdm(range(len(dataset))):
    #         item = dataset.__getitem__(index)
    #         label = item['y'].cpu()
    #         builtup_count += torch.sum(label).item()
    #
    #     background_count = len(dataset) * 256**2 - builtup_count
    #     container['builtup'].append(builtup_count)
    #     container['background'].append(background_count)

    container = {'builtup': [42179431.0, 17111600.0, 1629496.0], 'background': [228353177.0, 87090640.0, 2302664.0]}
    print(container)
    y_pos = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.barh(y_pos, container['background'], width, label='background')
    ax.barh(y_pos, container['builtup'], width, label='built-up',
            left=container['background'])

    # ax.set_xscale('log')
    ax.set_xlabel('Number of Pixels')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    ax.legend()
    plt.show()


if __name__ == '__main__':
    plot_number_of_pixels()
    pass