import sys
from pathlib import Path
import os
import timeit

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb

from networks.network_loader import create_network

from utils.datasets import UrbanExtractionDataset
from utils.augmentations import *
from utils.loss import get_criterion
from utils.evaluation import model_evaluation, model_testing

from experiment_manager.args import default_argument_parser
from experiment_manager.config import config


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

    sar_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    optical_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    fusion_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)

    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    # reset the generators
    dataset = UrbanExtractionDataset(cfg=cfg, dataset='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
    thresholds = torch.linspace(0, 1, 101)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        sar_loss_set, optical_loss_set, fusion_loss_set = [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x_fusion = batch['x'].to(device)
            y_gts = batch['y'].to(device)

            sar_logits, optical_logits, fusion_logits = net(x_fusion)

            sar_loss = sar_criterion(sar_logits, y_gts)
            sar_loss_set.append(sar_loss.item())

            optical_loss = optical_criterion(optical_logits, y_gts)
            optical_loss_set.append(optical_loss.item())

            fusion_loss = fusion_criterion(fusion_logits, y_gts)
            fusion_loss_set.append(fusion_loss.item())

            loss = sar_loss + optical_loss + fusion_loss
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                train_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'training', epoch_float, global_step,
                                                  max_samples=1_000)
                _ = model_evaluation(net, cfg, device, thresholds, 'validation', epoch_float, global_step,
                                     specific_index=train_argmaxF1, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'sar_loss': np.mean(sar_loss_set),
                    'optical_loss': np.mean(optical_loss_set),
                    'fusion_loss': np.mean(fusion_loss_set),
                    'labeled_percentage': 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                sar_loss_set, optical_loss_set, fusion_loss_set = [], [], []

            if cfg.DEBUG:
                break
            # end of batch

        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{cfg.NAME}_{epoch}.pkl'
            torch.save(net.state_dict(), net_file)
            # logs to load network
            train_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'training', epoch_float, global_step)
            validation_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'validation', epoch_float, global_step)

            wandb.log({
                'net_checkpoint': epoch,
                'checkpoint_step': global_step,
                'train_threshold': train_argmaxF1 / 100,
                'validation_threshold': validation_argmaxF1 / 100
            })
            model_testing(net, cfg, device, validation_argmaxF1, global_step, epoch_float)


if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = config.setup(args)

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
        train_net(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)