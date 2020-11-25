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

from networks.network_loader import create_network

from utils.datasets import MTUrbanExtractionDataset
from utils.augmentations import *
from utils.metrics import MultiThresholdMetric
from utils.loss import get_criterion
from utils.evaluation import model_evaluation, model_testing

from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config

from tqdm import tqdm


def train_mean_teacher(net, cfg):
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

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    supervised_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)

    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    student_net = net
    teacher_net = create_teacher_net(student_net, cfg)
    student_net.to(device)
    teacher_net.to(device)
    consistency_criterion = get_criterion(cfg.CONSISTENCY_TRAINER.CONSISTENCY_LOSS_TYPE)

    dataset = MTUrbanExtractionDataset(cfg=cfg, dataset='training')
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
        supervised_loss_set = []
        consistency_loss_set = []

        student_net.train()

        for i, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            x_student = batch['x_student'].to(device)
            x_teacher = batch['x_teacher'].to(device)
            y_gts = batch['y'].to(device)
            is_labeled = batch['is_labeled']

            logits_student = student_net(x_student)
            logits_teacher = teacher_net(x_teacher)

            supervised_loss, consistency_loss = None, None

            if is_labeled.any():
                supervised_loss = supervised_criterion(logits_student[is_labeled, ], y_gts[is_labeled])
                supervised_loss_set.append(supervised_loss.item())

            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                probs_teacher = torch.sigmoid(logits_teacher)
                consistency_loss = consistency_criterion(logits_student[not_labeled, ], probs_teacher[not_labeled, ])
                consistency_loss_set.append(consistency_loss.item())

            if supervised_loss is None and consistency_loss is not None:
                loss = cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
            elif supervised_loss is not None and consistency_loss is not None:
                loss = supervised_loss + cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
            else:
                loss = supervised_loss

            loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            global_step += 1

            # update teacher after each step
            update_teacher_net(student_net, teacher_net, alpha=1, global_step=global_step)

            if cfg.DEBUG:
                break
            # end of batch

        stop = timeit.default_timer()
        time_per_epoch = stop - start
        max_mem, max_cache = gpu_stats()

        if not cfg.DEBUG:
            wandb.log({
                'avg_loss': np.mean(loss_set),
                'avg_supervised_loss': np.mean(supervised_loss_set),
                'avg_consistency_loss': np.mean(consistency_loss_set),
                'gpu_memory': max_mem,
                'time': time_per_epoch,
                'step': global_step,
            })

        # evaluation on sample of training and validation set after ever epoch
        thresholds = torch.linspace(0, 1, 101)
        train_argmaxF1 = model_evaluation(teacher_net, cfg, device, thresholds, 'training', epoch,
                                                       global_step, max_samples=1_000)
        _ = model_evaluation(teacher_net, cfg, device, thresholds, 'validation', epoch, global_step,
                             specific_index=train_argmaxF1, max_samples=1_000)

        # updating best validation f1 score
        if epoch in save_checkpoints:
            print(f'saving network', flush=True)
            net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{cfg.NAME}_{epoch}.pkl'
            torch.save(teacher_net.state_dict(), net_file)


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

    # TODO: might not be necessary -> remove
    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg


def update_teacher_net(net, ema_net, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # TODO: fix warning due to add_
    for ema_param, param in zip(ema_net.parameters(), net.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    return ema_net


def create_teacher_net(net, cfg):
    net_copy = type(net)(cfg)  # get a new instance
    net_copy.load_state_dict(net.state_dict())  # copy weights and stuff
    # TODO: not sure about this one
    for param in net_copy.parameters():
        param.detach_()
    return net_copy


if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = setup(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = create_network(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)

    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='urban_extraction',
            tags=['run', 'urban', 'extraction', 'segmentation', ],
        )

    try:
        train_mean_teacher(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
