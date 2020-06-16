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

from networks.network_loader import load_network

from utils.dataloader import UrbanExtractionDataset
from utils.augmentations import *
from utils.metrics import MultiThresholdMetric
from utils.loss import criterion_from_cfg

from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config


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

    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.0005)
    criterion = criterion_from_cfg(cfg)

    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    net.to(device)
    global_step = 0
    epochs = cfg.TRAINER.EPOCHS

    use_edge_loss = cfg.MODEL.LOSS_TYPE == 'FrankensteinEdgeLoss'

    # reset the generators
    dataset = UrbanExtractionDataset(cfg=cfg, dataset='train')
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

    best_test_f1 = 0  # used to save network
    for epoch in range(epochs):
        start = timeit.default_timer()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0

        net.train()

        loss_set = []
        positive_pixels_set = []  # Used to evaluated image over sampling techniques
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            image_weight = batch['image_weight']

            y_pred = net(x)

            if cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
                # y_pred = y_pred # Cross entropy loss doesn't like single channel dimension
                y_gts = y_gts.long()  # Cross entropy loss requires a long as target
            if use_edge_loss:
                edge_mask = y_gts[:, [-1]]
                y_gts = y_gts[:, [0]]
                loss = criterion(y_pred, y_gts, edge_mask, cfg.TRAINER.EDGE_LOSS_SCALE)
            else:
                loss = criterion(y_pred, y_gts)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            positive_pixels_set.extend(image_weight.cpu().numpy())

            global_step += 1

        stop = timeit.default_timer()
        time_per_epoch = stop - start

        max_mem, max_cache = gpu_stats()
        print(f'step {global_step},  avg loss: {np.mean(loss_set):.4f}, cuda mem: {max_mem} MB, cuda cache: {max_cache} MB, time: {time_per_epoch:.2f}s', flush=True)

        if not cfg.DEBUG:
            wandb.log({
                'loss': np.mean(loss_set),
                'gpu_memory': max_mem,
                'time': time_per_epoch,
                'total_positive_pixels': np.mean(positive_pixels_set),
                'step': global_step,
            })

        loss_set = []
        positive_pixels_set = []
        start = stop

        # evaluation on sample of train and test set after ever epoch
        test_f1 = model_eval(net, cfg, device, run_type='test', step=global_step, epoch=epoch)
        train_f1 = model_eval(net, cfg, device, max_samples=100, run_type='train', step=global_step, epoch=epoch)

        if test_f1 > best_test_f1:
            print(f'BEST PERFORMANCE SO FAR!', flush=True)
            best_test_f1 = test_f1

            if cfg.SAVE_MODEL and not cfg.DEBUG:
                print(f'saving network', flush=True)
                model_name = 'best_net.pkl'
                save_path = os.path.join(cfg.OUTPUT_DIR, model_name)
                torch.save(net.state_dict(), save_path)


def image_sampling_weight(samples_metadata):
    print('performing oversampling...', end='', flush=True)
    empty_image_baseline = 1000
    sampling_weights = np.array([float(sample['img_weight']) for sample in samples_metadata]) + empty_image_baseline
    print('done', flush=True)
    return sampling_weights


# TODO: move to utils
def model_eval(net, cfg, device, run_type='test', max_samples=1000, step=0, epoch=0):

    F1_THRESH = torch.linspace(0, 1, 100).to(device)
    y_true_set = []
    y_pred_set = []

    measurer = MultiThresholdMetric(F1_THRESH)

    def evaluate(y_true, y_pred):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        y_true_set.append(y_true.cpu())
        y_pred_set.append(y_pred.cpu())

        measurer.add_sample(y_true, y_pred)

    dataset = UrbanExtractionDataset(cfg=cfg, dataset=run_type)
    inference_loop(net, cfg, device, evaluate, max_samples=max_samples, dataset=dataset)

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1 = measurer.compute_f1()
    fpr, fnr = measurer.compute_basic_metrics()
    maxF1 = f1.max()
    argmaxF1 = f1.argmax()
    best_fpr = fpr[argmaxF1]
    best_fnr = fnr[argmaxF1]
    print(f'{maxF1.item():.3f}', flush=True)

    if not cfg.DEBUG:
        wandb.log({f'{run_type} max F1': maxF1,
                   f'{run_type} argmax F1': argmaxF1,
                   # f'{set_name} Average Precision': ap,
                   f'{run_type} false positive rate': best_fpr,
                   f'{run_type} false negative rate': best_fnr,
                   'step': step,
                   'epoch': epoch,
                   })

    return maxF1.item()


def inference_loop(net, cfg, device, callback=None, batch_size=1, run_type='test', max_samples=999999999,
                   dataset=None, callback_include_x=False):

    net.to(device)
    net.eval()

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                       shuffle=cfg.DATALOADER.SHUFFLE, drop_last=True)
    # dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=cfg.DATALOADER.SHUFFLE, drop_last=True)

    dataset_length = np.minimum(len(dataset), max_samples)
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            imgs = batch['x'].to(device)
            y_label = batch['y'].to(device)

            y_pred = net(imgs)

            if step % 100 == 0 or step == dataset_length-1:
                # print(f'Processed {step+1}/{dataset_length}')
                pass

            if y_pred.shape[1] > 1:  # multi-class
                # In Two class Cross entropy mode, positive classes are in Channel #2
                y_pred = torch.softmax(y_pred, dim=1)
            else:
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

    if args.log_dir: # Override Output dir
        cfg.OUTPUT_DIR = path.join(args.log_dir, args.config_file)
    else:
        cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, args.config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg


if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = setup(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = load_network(cfg)

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
        train_net(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
