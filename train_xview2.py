import sys
from os import path
import os
from argparse import ArgumentParser
import datetime
import enum
import timeit

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data
from torch.nn import functional as F
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from tabulate import tabulate
import wandb

from debug_tools import __benchmark_init, benchmark
from unet import UNet
from unet.dataloader import Xview2Detectron2Dataset
from unet.augmentations import *

from experiment_manager.metrics import f1_score
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
from experiment_manager.loss import soft_dice_loss, soft_dice_loss_balanced, jaccard_like_loss, jaccard_like_balanced_loss
from eval_unet_xview2 import model_eval

# import hp

def train_net(net,
              cfg):

    log_path = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_path)

    run_config = {}
    run_config['CONFIG_NAME'] = cfg.NAME
    run_config['device'] = device
    run_config['log_path'] = cfg.OUTPUT_DIR
    run_config['training_set'] = cfg.DATASETS.TRAIN
    run_config['test set'] = cfg.DATASETS.TEST
    run_config['epochs'] = cfg.TRAINER.EPOCHS
    run_config['learning rate'] = cfg.TRAINER.LR
    run_config['batch size'] = cfg.TRAINER.BATCH_SIZE
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))


    optimizer = optim.Adam(net.parameters(),
                          lr=cfg.TRAINER.LR,
                          weight_decay=0.0005)
    if cfg.MODEL.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
        balance_weight = [cfg.MODEL.NEGATIVE_WEIGHT, cfg.MODEL.POSITIVE_WEIGHT]
        balance_weight = torch.tensor(balance_weight).float().to(device)
        criterion = nn.CrossEntropyLoss(weight = balance_weight)
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceLoss':
        criterion = soft_dice_loss
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceBalancedLoss':
        criterion = soft_dice_loss_balanced
    elif cfg.MODEL.LOSS_TYPE == 'JaccardLikeLoss':
        criterion = jaccard_like_loss
    elif cfg.MODEL.LOSS_TYPE == 'ComboLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'WeightedComboLoss':
        criterion = lambda pred, gts: 2 * F.binary_cross_entropy_with_logits(pred, gts) + soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'FrankensteinLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + jaccard_like_balanced_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'FrankensteinEdgeLoss':
        criterion = FrankensteinEdgeLoss


    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    __benchmark_init()
    global_step = 0
    epochs = cfg.TRAINER.EPOCHS

    use_edge_loss = cfg.MODEL.LOSS_TYPE == 'FrankensteinEdgeLoss'

    trfm = []
    trfm.append(BGR2RGB())
    if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
        trfm.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    trfm.append(Npy2Torch())
    if cfg.AUGMENTATION.ENABLE_VARI: trfm.append(VARI())
    trfm = transforms.Compose(trfm)

    # reset the generators
    dataset = Xview2Detectron2Dataset(cfg.DATASETS.TRAIN[0],
                                      pre_or_post=cfg.DATASETS.PRE_OR_POST,
                                      include_image_weight=True,
                                      transform=trfm,
                                      include_edge_mask=use_edge_loss,
                                      )

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }

    # sampler
    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'simple':
        image_p = image_sampling_weight(dataset.dataset_metadata)
        sampler = torch_data.WeightedRandomSampler(weights=image_p, num_samples=len(image_p))
        dataloader_kwargs['sampler'] = sampler
        dataloader_kwargs['shuffle'] = False

    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)


    for epoch in range(epochs):
        start = timeit.default_timer()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0

        net.train()
        # mean AP, mean AUC, max F1
        mAP_set_train, mAUC_set_train, maxF1_train = [],[],[]
        loss_set, f1_set = [], []
        positive_pixels_set = [] # Used to evaluated image over sampling techniques
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            image_weight = batch['image_weight']


            y_pred = net(x)

            if cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
                # y_pred = y_pred # Cross entropy loss doesn't like single channel dimension
                y_gts = y_gts.long()# Cross entropy loss requires a long as target

            if use_edge_loss:
                # TODO
                edge_mask = y_gts[:,[-1]]
                y_gts = y_gts[:,[0]]
                loss = criterion(y_pred, y_gts, edge_mask, cfg.TRAINER.EDGE_LOSS_SCALE)
            else:
                loss = criterion(y_pred, y_gts)


            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            positive_pixels_set.extend(image_weight.cpu().numpy())

            if global_step % 100 == 0 or global_step == 0:
                # time per 100 steps
                stop = timeit.default_timer()
                time_per_n_batches= stop - start

                if global_step % 10000 == 0 and global_step > 0:
                    check_point_name = f'cp_{global_step}.pkl'
                    save_path = os.path.join(log_path, check_point_name)
                    torch.save(net.state_dict(), save_path)

                # Averaged loss and f1 writer

                # writer.add_scalar('f1/train', np.mean(f1_set), global_step)

                max_mem, max_cache = gpu_stats()
                print(f'step {global_step},  avg loss: {np.mean(loss_set):.4f}, cuda mem: {max_mem} MB, cuda cache: {max_cache} MB, time: {time_per_n_batches:.2f}s',
                      flush=True)

                wandb.log({
                    'loss': np.mean(loss_set),
                    'gpu_memory': max_mem,
                    'time': time_per_n_batches,
                    'total_positive_pixels': np.mean(positive_pixels_set),
                    'step': global_step,
                })

                loss_set = []
                positive_pixels_set = []

                start = stop

            # torch.cuda.empty_cache()
            global_step += 1

        if epoch % 2 == 0:
            # Evaluation after every other epoch
            model_eval(net, cfg, device, max_samples=100, step=global_step, epoch=epoch)
            model_eval(net, cfg, device, max_samples=100, run_type='TRAIN', step=global_step, epoch=epoch)

def FrankensteinEdgeLoss(p, y, neg_edge_mask, lambda_factor=1):
    p2 = torch.sigmoid(p).clamp(1e-7)
    ce = y * p2.log() + (1-y) * (1-p2).log() * neg_edge_mask
    ce_loss = -ce.mean() * lambda_factor

    # loss2 = y - p >= 0
    # ce_loss_reference = F.binary_cross_entropy_with_logits(p, y, reduction='none')
    F.nll_loss()
    loss = ce_loss + jaccard_like_balanced_loss(p, y)
    return loss

def image_sampling_weight(dataset_metadata):
    print('performing oversampling...', end='', flush=True)
    EMPTY_IMAGE_BASELINE = 1000
    image_p = np.array([image_desc['pre']['image_weight'] for image_desc in dataset_metadata]) + EMPTY_IMAGE_BASELINE
    print('done', flush=True)
    # normalize to [0., 1.]
    image_p = image_p
    return image_p

class LULC(enum.Enum):
    BACKGROUND = (0, 'Background', 'black')
    NO_DATA = (1, 'No Data', 'white')
    NO_DAMAGE = (2, 'No damage', 'xkcd:lime')
    MINOR_DAMAGE = (3, 'Minor Damage', 'yellow')
    MAJOR_DAMAGE = (4, 'Major Damage', 'orange')
    DESTROYED = (5, 'Destroyed', 'red')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3

lulc_cmap = ListedColormap([entry.color for entry in LULC])
lulc_norm = BoundaryNorm(np.arange(-0.5, 6, 1), lulc_cmap.N)

def gpu_stats():
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e6 # bytes to MB
    max_memory_cached = torch.cuda.max_memory_cached() /1e6


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

    out_channels = cfg.MODEL.OUT_CHANNELS
    net = UNet(cfg)

    if args.resume and args.resume_from:
        full_model_path = path.join(cfg.OUTPUT_DIR, args.model_path)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        project='urban_dl',
        tags=['run', 'localization'],
    )
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        train_net(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


