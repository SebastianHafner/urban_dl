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
from experiment_manager.config import new_config


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

    dataset = UrbanExtractionDataset(cfg=cfg, dataset='training', include_unlabeled=False)
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
    steps_per_epoch = len(dataset) // cfg.TRAINER.BATCH_SIZE

    # tracking variables
    global_step = 0

    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}.')

        start = timeit.default_timer()
        loss_set, supervised_loss_set, consistency_loss_set = [], [], []
        n_labeled, n_notlabeled = 0, 0

        student_net.train()
        teacher_net.eval()

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            is_labeled = batch['is_labeled']

            logits_student = student_net(x)
            logits_teacher = teacher_net(x)
            logits_teacher = logits_teacher.detach()

            supervised_loss, consistency_loss = None, None

            if is_labeled.any():
                supervised_loss = supervised_criterion(logits_student[is_labeled, ], y_gts[is_labeled])
                supervised_loss_set.append(supervised_loss.item())
                n_labeled += torch.sum(is_labeled).item()

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
            epoch_float = global_step / steps_per_epoch

            # update teacher after each step
            update_teacher_net(student_net, teacher_net, alpha=1, global_step=global_step)

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                thresholds = torch.linspace(0, 1, 101)
                train_argmaxF1 = model_evaluation(student_net, cfg, device, thresholds, 'training', epoch_float,
                                                  global_step, max_samples=1_000)
                _ = model_evaluation(student_net, cfg, device, thresholds, 'validation', epoch_float, global_step,
                                     specific_index=train_argmaxF1, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                labeled_percentage = n_labeled / (n_labeled + n_notlabeled)
                wandb.log({
                    'loss': np.mean(loss_set),
                    'supervised_loss': 0 if len(supervised_loss_set) == 0 else np.mean(supervised_loss_set),
                    'consistency_loss': 0 if len(consistency_loss_set) == 0 else np.mean(consistency_loss_set),
                    'labeled_percentage': labeled_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })

                # resetting stuff
                start = timeit.default_timer()
                n_labeled, n_notlabeled = 0, 0

            if cfg.DEBUG:
                break
            # end of batch

        # saving network
        if epoch in save_checkpoints and not cfg.DEBUG:
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
