import sys
from os import path
from pathlib import Path
import os
from argparse import ArgumentParser
import datetime
import enum
import timeit

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb

from networks.network_loader import create_network, load_network

from utils.datasets import STUrbanExtractionDataset
from utils.augmentations import *
from utils.evaluation import model_evaluation, model_testing
from utils.loss import get_criterion

from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config

from tqdm import tqdm


def train_sar_teacher(cfg, sar_cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = create_network(cfg)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    net.to(device)

    sar_net_file = Path(sar_cfg.OUTPUT_BASE_DIR) / f'{sar_cfg.NAME}_{sar_cfg.INFERENCE.CHECKPOINT}.pkl'
    sar_net = load_network(sar_cfg, sar_net_file)
    sar_net.to(device)
    sar_net.eval()
    consistency_criterion = get_criterion(cfg.CONSISTENCY_TRAINER.CONSISTENCY_LOSS_TYPE)

    dataset = STUrbanExtractionDataset(cfg=cfg, sar_cfg=sar_cfg, run_type='training')
    print(dataset)
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS

    # tracking variables
    global_step = 0

    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []
        sar_loss_set = []
        consistency_loss_set = []

        net.train()

        for i, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            x_sar = batch['x_sar'].to(device)
            y_gts = batch['y'].to(device)
            is_labeled = batch['is_labeled']

            logits = net(x)

            loss, consistency_loss = None, None

            if is_labeled.any():
                loss = criterion(logits[is_labeled, ], y_gts[is_labeled])
                loss_set.append(loss.item())

            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                with torch.no_grad():
                    probs_sar = torch.sigmoid(sar_net(x_sar[not_labeled, ]))
                    if cfg.CONSISTENCY_TRAINER.APPLY_THRESHOLD:
                        output_sar = (probs_sar > sar_cfg.INFERENCE.THRESHOLDS.VALIDATION).float()
                    else:
                        output_sar = probs_sar
                # mean square error
                consistency_loss = consistency_criterion(logits[not_labeled, ], output_sar)
                consistency_loss_set.append(consistency_loss.item())

            if loss is None and consistency_loss is not None:
                loss = cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
            elif loss is not None and consistency_loss is not None:
                loss = loss + cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
            else:
                loss = loss

            loss.backward()
            optimizer.step()

            global_step += 1

            if cfg.DEBUG:
                break
            # end of batch

        stop = timeit.default_timer()
        time_per_epoch = stop - start
        max_mem, max_cache = gpu_stats()

        if not cfg.DEBUG:
            wandb.log({
                'avg_loss': np.mean(loss_set),
                'avg_loss_sar': np.mean(sar_loss_set),
                'avg_consistency_loss': np.mean(consistency_loss_set),
                'gpu_memory': max_mem,
                'time': time_per_epoch,
                'step': global_step,
                'epoch': epoch,
            })

        # evaluation on sample of training and validation set after ever epoch
        thresholds = torch.linspace(0, 1, 101)
        train_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'training', epoch, global_step,
                                          max_samples=10_000)
        val_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'validation', epoch, global_step,
                                        specific_index=train_argmaxF1, max_samples=10_000)
        model_testing(net, cfg, device, val_argmaxF1, epoch, global_step)

        if epoch in save_checkpoints:
            print(f'saving network', flush=True)
            net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{cfg.NAME}_{epoch}.pkl'
            torch.save(net.state_dict(), net_file)


def image_sampling_weight(samples_metadata):
    print('performing oversampling...', end='', flush=True)
    empty_image_baseline = 1000
    sampling_weights = np.array([float(sample['img_weight']) for sample in samples_metadata]) + empty_image_baseline
    print('done', flush=True)
    return sampling_weights




def gpu_stats():
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e6  # bytes to MB
    max_memory_cached = torch.cuda.max_memory_cached() / 1e6
    return int(max_memory_allocated), int(max_memory_cached)


def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file

    sar_cfg = new_config()
    sar_cfg.merge_from_file(f'configs/{args.sar_config_file}.yaml')
    sar_cfg.merge_from_list(args.opts)
    sar_cfg.NAME = args.sar_config_file

    return cfg, sar_cfg


if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg, sar_cfg = setup(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='urban_extraction',
            tags=['run', 'urban', 'extraction', 'segmentation', ],
        )

    try:
        train_sar_teacher(cfg, sar_cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)