import sys
import os
from optparse import OptionParser
import datetime
from collections import OrderedDict
import enum

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data
from tensorboardX import SummaryWriter
from coolname import generate_slug
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from eval import eval_net
from unet import UNet
from unet.utils import SloveniaDataset

DATASET_DIR = 'data/slovenia/slovenia2017.hdf5'

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              num_dataloaders = 1,
              device=torch.device('cpu'),
              img_scale=0.5):

    run_name = datetime.datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
    log_path = 'logs/%s' % run_name
    writer = SummaryWriter(log_path)

    # TODO Save Run Config in Pandas
    # TODO Save
    print('>>>>> run name', run_name)
    print('run started, running on ', device)
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss()

    net.to(device)

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        dataset = SloveniaDataset(DATASET_DIR, epoch)
        dataloader = torch_data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           pin_memory=True,
                                           num_workers=num_dataloaders,
                                           drop_last=True,
                                           )

        epoch_loss = 0
        datasize = dataset.length

        for i, (imgs, true_masks) in enumerate(dataloader):
            global_step = epoch * datasize + i
            print('step', global_step)
            print('data sample loaded')
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            print('data sample loaded to CUDA')
            masks_pred = net(imgs)
            print('inference computed')
            loss = criterion(masks_pred, true_masks)
            print('loss computed ')
            epoch_loss += loss.item()

            print('loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            print('loss backward')
            optimizer.step()
            print('optimzer executed')

            # Write things in
            if global_step % 30 == 0:
                writer.add_scalar('loss', loss, global_step)
                print('add loss')
                visualize_image(imgs, masks_pred, true_masks, writer, global_step)
                print('visualize image')



class LULC(enum.Enum):
    NO_DATA = (0, 'No Data', 'white')
    CULTIVATED_LAND = (1, 'Cultivated Land', 'xkcd:lime')
    FOREST = (2, 'Forest', 'xkcd:darkgreen')
    GRASSLAND = (3, 'Grassland', 'orange')
    SHRUBLAND = (4, 'Shrubland', 'xkcd:tan')
    WATER = (5, 'Water', 'xkcd:azure')
    WETLAND = (6, 'Wetlands', 'xkcd:lightblue')
    TUNDRA = (7, 'Tundra', 'xkcd:lavender')
    ARTIFICIAL_SURFACE = (8, 'Artificial Surface', 'crimson')
    BARELAND = (9, 'Bareland', 'xkcd:beige')
    SNOW_AND_ICE = (10, 'Snow and Ice', 'black')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3

lulc_cmap = ListedColormap([entry.color for entry in LULC])

def visualize_image(input_image, output_segmentation, gt_segmentation, writer:SummaryWriter, global_step):

    # TODO This is slow, consider making this working in a background thread. Or making the entire tensorboardx work in a background thread

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig.tight_layout()
    plt.tight_layout()
    fig.set_figheight(5)
    fig.set_figwidth(15)

    # Plot image
    img = toNp(input_image)[...,3]  # first item, B channel only
    ax0.imshow(img)
    ax0.axis('off')

    # Plot segments
    out_seg = toNp(output_segmentation)
    out_seg_argmax = np.argmax(out_seg, axis=-1)

    ax1.imshow(out_seg_argmax.squeeze(), cmap = lulc_cmap)
    ax1.set_title('output')
    ax1.axis('off')

    # plot ground truth
    gt = toNp_vanilla(gt_segmentation)
    ax2.imshow(gt.squeeze(), cmap=lulc_cmap)
    ax2.set_title('ground_truth')
    ax2.axis('off')

    writer.add_figure('output_image',fig,global_step)


def toNp_vanilla(t:torch.Tensor):
    return t[0,...].detach().cpu().numpy()

def toNp(t:torch.Tensor):
    # Pick the first item
    return to_H_W_C(t)[0,...].detach().cpu().numpy()

def to_C_H_W(t:torch.Tensor):
    # From [B, H, W, C] to [B, C, H, W]
    assert t.shape[1] == t.shape[2] and t.shape[3] != t.shape[2], 'are you sure this tensor is in [B, H, W, C] format?'
    return t.permute(0,3,1,2)

def to_H_W_C(t:torch.Tensor):
    # From [B, C, H, W] to [B, H, W, C]
    assert t.shape[1] != t.shape[2], 'are you sure this tensor is in [B, C, H, W] format?'
    return t.permute(0,2,3,1)


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-w', '--num-worker', dest='num_dataloaders', default=1,
                      type='int', help='number of dataloader workers')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=6, n_classes=10)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    else:
        device = torch.device('cpu')
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  num_dataloaders = args.num_dataloaders,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
