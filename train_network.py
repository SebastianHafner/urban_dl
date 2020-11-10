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

from utils.dataloader import UrbanExtractionDataset
from utils.augmentations import *
from utils.metrics import MultiThresholdMetric
from utils.loss import criterion_from_cfg

from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config

from tqdm import tqdm

def train_net(net, cfg):

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
    criterion = criterion_from_cfg(cfg)

    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    use_edge_loss = cfg.MODEL.LOSS_TYPE == 'FrankensteinEdgeLoss'

    # reset the generators
    dataset = UrbanExtractionDataset(cfg=cfg, dataset='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }

    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'simple':
        image_p = image_sampling_weight(dataset.samples)
        sampler = torch_data.WeightedRandomSampler(weights=image_p, num_samples=len(image_p))
        dataloader_kwargs['sampler'] = sampler
        dataloader_kwargs['shuffle'] = False
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS

    # tracking variables
    global_step = 0
    best_val_f1 = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}.')

        start = timeit.default_timer()
        epoch_loss = 0
        loss_set = []
        positive_pixels_set = []  # Used to evaluated image over sampling techniques

        net.train()

        for i, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            image_weight = batch['image_weight']

            y_pred = net(x)

            loss = criterion(y_pred, y_gts)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            positive_pixels_set.extend(image_weight.cpu().numpy())

            global_step += 1
            if cfg.DEBUG:
                break
            # end of batch

        stop = timeit.default_timer()
        time_per_epoch = stop - start
        max_mem, max_cache = gpu_stats()

        if not cfg.DEBUG:
            wandb.log({
                'loss': loss.item(),
                'avg_loss': np.mean(loss_set),
                'gpu_memory': max_mem,
                'time': time_per_epoch,
                'total_positive_pixels': np.mean(positive_pixels_set),
                'step': global_step,
            })

        # evaluation on sample of training and validation set after ever epoch
        thresholds = torch.linspace(0, 1, 101)
        train_maxF1, train_argmaxF1 = model_eval(net, cfg, device, thresholds, 'training', epoch, global_step,
                                                 max_samples=500)
        val_f1, val_argmaxF1 = model_eval(net, cfg, device, thresholds, 'validation', epoch, global_step,
                                          specific_index=train_argmaxF1, max_samples=500)

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
                                       shuffle=cfg.DATALOADER.SHUFFLE, drop_last=True)

    dataset_length = np.minimum(len(dataset), max_samples)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            imgs = batch['x'].to(device)
            y_label = batch['y'].to(device)

            y_pred = net(imgs)

            y_pred = torch.sigmoid(y_pred)

            if callback:
                if callback_include_x:
                    callback(imgs, y_label, y_pred)
                else:
                    callback(y_label, y_pred)

            if (max_samples is not None) and step >= max_samples:
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

    # TODO: might not be necessary -> remove
    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg


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
            project='urban_extraction_version4',
            tags=['run', 'urban', 'extraction', 'segmentation', ],
        )

    try:
        train_net(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
