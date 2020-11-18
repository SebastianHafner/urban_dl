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

from utils.dataloader import UrbanExtractionDataset, STUrbanExtractionDataset
from utils.augmentations import *
from utils.metrics import MultiThresholdMetric
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
    consistency_criterion = get_criterion(cfg.CONSISTENCY_TRAINER.LOSS_TYPE)

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
    best_val_f1 = 0

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
                output = torch.sigmoid(logits[not_labeled, ])
                with torch.no_grad():
                    probs_sar = torch.sigmoid(sar_net(x_sar[not_labeled, ]))
                    if cfg.CONSISTENCY_TRAINER.APPLY_THRESHOLD:
                        output_sar = (probs_sar > sar_cfg.INFERENCE.THRESHOLDS.VALIDATION).float()
                    else:
                        output_sar = probs_sar
                # mean square error
                # consistency_loss = torch.div(torch.sum(torch.pow(output - output_sar, 2)), torch.numel(output))
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
        train_maxF1, train_argmaxF1 = model_eval(net, cfg, device, thresholds, 'training', epoch, global_step,
                                                 max_samples=10_000)
        val_f1, val_argmaxF1 = model_eval(net, cfg, device, thresholds, 'validation', epoch, global_step,
                                          specific_index=train_argmaxF1, max_samples=10_000)

        # TODO: add evaluation on test set

        # updating best validation f1 score
        best_val_f1 = val_f1 if val_f1 > best_val_f1 else val_f1

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


# specific threshold creates an additional log for that threshold
# can be used to apply best training threshold to validation set
def model_eval(net, cfg, device, thresholds: torch.Tensor, run_type: str, epoch: int, step: int,
               max_samples: int = 1000, specific_index: int = None):
    y_true_set = []
    y_pred_set = []

    thresholds = thresholds.to(device)
    measurer = MultiThresholdMetric(thresholds)

    def evaluate(y_true, y_pred):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        y_true_set.append(y_true.cpu())
        y_pred_set.append(y_pred.cpu())

        measurer.add_sample(y_true, y_pred)

    dataset = UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True)
    inference_loop(net, cfg, device, evaluate, max_samples=max_samples, dataset=dataset)

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1s = measurer.compute_f1()
    precisions, recalls = measurer.precision, measurer.recall

    # best f1 score for passed thresholds
    f1 = f1s.max()
    argmax_f1 = f1s.argmax()

    best_thresh = thresholds[argmax_f1]
    precision = precisions[argmax_f1]
    recall = recalls[argmax_f1]

    print(f'{f1.item():.3f}', flush=True)

    if specific_index is not None:
        specific_f1 = f1s[specific_index]
        specific_thresh = thresholds[specific_index]
        specific_precision = precisions[specific_index]
        specific_recall = recalls[specific_index]
        if not cfg.DEBUG:
            wandb.log({f'{run_type} specific F1': specific_f1,
                       f'{run_type} specific threshold': specific_thresh,
                       f'{run_type} specific precision': specific_precision,
                       f'{run_type} specific recall': specific_recall,
                       'step': step, 'epoch': epoch,
                       })

    if not cfg.DEBUG:
        wandb.log({f'{run_type} F1': f1,
                   f'{run_type} threshold': best_thresh,
                   f'{run_type} precision': precision,
                   f'{run_type} recall': recall,
                   'step': step, 'epoch': epoch,
                   })

    return f1.item(), argmax_f1.item()


def inference_loop(net, cfg, device, callback=None, batch_size=1, max_samples=999999999,
                   dataset=None, callback_include_x=False):
    net.to(device)
    net.eval()

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                       shuffle=True, drop_last=True)

    dataset_length = np.minimum(len(dataset), max_samples)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            if step == dataset_length:
                break

            imgs = batch['x'].to(device)
            y_label = batch['y'].to(device)

            y_pred = net(imgs)
            y_pred = torch.sigmoid(y_pred)

            if callback:
                if callback_include_x:
                    callback(imgs, y_label, y_pred)
                else:
                    callback(y_label, y_pred)

            if cfg.DEBUG:
                break


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
