import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from utils.datasets import UrbanExtractionDataset, SpaceNet7Dataset
from experiment_manager.config import config
from utils.geotiff import *
import numpy as np
from tqdm import tqdm
import torch
import json

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def train_validation_statistics(config_name: str):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    training_dataset = UrbanExtractionDataset(cfg, 'training', no_augmentations=True)
    train_labeled = 0
    train_unlabeled = 0
    train_builtup = 0
    for index in tqdm(range(len(training_dataset))):
        item = training_dataset.__getitem__(index)
        label = item['y'].cpu()
        if item['is_labeled']:
            train_labeled += torch.numel(label)
            train_builtup += torch.sum(label).item()
        else:
            train_unlabeled += torch.numel(label)
    train_background = train_labeled - train_builtup
    print(f'labeled: {train_labeled} (builtup: {train_builtup}, bg: {train_background}) - unlabeled: {train_unlabeled}')

    validation_dataset = UrbanExtractionDataset(cfg, 'validation', no_augmentations=True, include_unlabeled=False)
    val_labeled = 0
    val_builtup = 0
    for index in tqdm(range(len(validation_dataset))):
        item = training_dataset.__getitem__(index)
        label = item['y'].cpu()
        val_labeled += torch.numel(label)
        val_builtup += torch.sum(label).item()
    val_background = val_labeled - val_builtup
    print(f'labeled: {val_labeled} (builtup: {val_builtup}, bg: {val_background})')

    output_data = {
        'train_labeled': train_labeled,
        'train_builtup': train_builtup,
        'train_background': train_background,
        'train_unlabeled': train_unlabeled,
        'val_labeled': val_labeled,
        'val_builtup': val_builtup,
        'val_background': val_background
    }

    output_file = ROOT_PATH / 'plots' / 'dataset' / f'train_validation_statistics_{config_name}.json'
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def test_statistics(config_name: str):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    test_dataset = SpaceNet7Dataset(cfg)
    data = {}
    for index in tqdm(range(len(test_dataset))):
        item = test_dataset.__getitem__(index)
        label = item['y'].cpu()
        group_name = item['group_name']
        labeled = torch.numel(label)
        builtup = torch.sum(label).item()
        if not group_name in data.keys():
            data[group_name] = {
                'labeled': 0,
                'builtup': 0,
                'background': 0
            }
        data[group_name]['labeled'] += labeled
        data[group_name]['builtup'] += builtup
        data[group_name]['background'] += (labeled - builtup)

    output_file = ROOT_PATH / 'plots' / 'dataset' / f'test_statistics_{config_name}.json'
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def plot_train_validation(config_name: str):
    mpl.rcParams.update({'font.size': 14})
    data = load_json(ROOT_PATH / 'plots' / 'dataset' / f'train_validation_statistics_{config_name}.json')
    print(data)
    ypos = [0.5, 1, 1.5]
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 3))
    neg = ax.barh(ypos[-1], data['train_background'], width, label='Negative', color='lightgray', edgecolor='k')
    pos = ax.barh(ypos[-1], data['train_builtup'], width, label='Positive', left=data['train_background'],
                  color='darkblue', edgecolor='k')
    na = ax.barh(ypos[1], data['train_unlabeled'], width, label='N/A', color='white', edgecolor='k')
    ax.barh(ypos[0], data['val_background'], width, label='Negative', color='lightgray', edgecolor='k')
    ax.barh(ypos[0], data['val_builtup'], width, label='Positive', left=data['val_background'], color='darkblue',
            edgecolor='k')

    ax.ticklabel_format(style='sci')
    ax.set_xlabel('Number of Pixels')

    ax.set_yticks(ypos)
    ax.set_yticklabels(['Validation labeled', 'Train unlabeled', 'Train labeled'])
    ax.legend((neg, pos, na), ('Negative', 'Positive', 'N/A'), loc='lower right', ncol=3)
    plt.show()


def plot_test(config_name: str):
    mpl.rcParams.update({'font.size': 14})
    data = load_json(ROOT_PATH / 'plots' / 'dataset' / f'test_statistics_{config_name}.json')
    print(data)
    width = 0.25
    ypos = np.arange(len(data.keys()))
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (site, site_data) in enumerate(data.items()):
        site_data = data[site]
        neg = ax.barh(i, site_data['background'], width, label='Negative', color='lightgray', edgecolor='k')
        pos = ax.barh(i, site_data['builtup'], width, label='Positive', left=site_data['background'],
                      color='darkblue', edgecolor='k')

    ax.ticklabel_format(style='sci')
    ax.set_xlabel('Number of Pixels')

    ax.set_yticks(ypos)
    ax.set_yticklabels(data.keys())
    ax.legend((neg, pos), ('Negative', 'Positive'), loc='lower right', ncol=1)
    plt.show()

if __name__ == '__main__':
    config_name = 'fusiondual_semisupervised_extended'
    # train_validation_statistics(config_name)
    # plot_train_validation(config_name)
    # test_statistics(config_name)
    plot_test(config_name)