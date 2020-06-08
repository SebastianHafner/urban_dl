from tensorflow_implementation.model import get_unet
from experiment_manager.config import config

from tensorflow.python.keras import optimizers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import losses


if __name__ == '__main__':

    optim = tf.keras.optimizers.Adam(
        learning_rate=cfg.TRAINER.LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam'
    )

    config.load_cfg(

    model = get_unet(cfg)

    model.compile(
        optimizer=optim,
        loss=losses.get(LOSS),
        metrics=[metrics.get(metric) for metric in METRICS]
    )