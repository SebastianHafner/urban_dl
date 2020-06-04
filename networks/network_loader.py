import segmentation_models_pytorch as smp


from networks.unet import UNet, DualStreamUNet
from networks.resnet import ResNet


def load_network(cfg):

    architecture = cfg.MODEL.TYPE

    if architecture == 'unet':

        if cfg.MODEL.BACKBONE.ENABLED:
            net = smp.Unet(
                cfg.MODEL.BACKBONE.TYPE,
                encoder_weights=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS,
                in_channels=cfg.MODEL.IN_CHANNELS,
                classes=cfg.MODEL.OUT_CHANNELS,
                activation=None,
            )
        else:
            net = UNet(cfg)

    elif architecture == 'dualstreamunet':
        net = DualStreamUNet(cfg)

    else:
        net = UNet(cfg)

    return net
