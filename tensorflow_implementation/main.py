import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics

from tensorflow_implementation.data_generator import DataGenerator
from tensorflow_implementation.model import unet

from pathlib import Path
import matplotlib.pyplot as plt
import os
from os import path
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config


def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'../configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file

    if args.log_dir:  # Override Output dir
        cfg.OUTPUT_DIR = path.join(args.log_dir, args.config_file)
    else:
        cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, args.config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg


class DisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self, data_generator: DataGenerator, indices: list = []):
        self.data_generator = data_generator
        self.indices = indices

    def on_epoch_end(self, epoch, logs=None):

        title = f'Predictions after {epoch + 1} epochs'

        for i in self.indices:
            self.display_sample(i)

    def display_sample(index: int):
        img, label = self.data_generator.
        pass


# Segmentation tutorial https://www.tensorflow.org/tutorials/images/segmentation
if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = setup(args)

    # Generators
    training_generator = DataGenerator(cfg, 'train')
    validation_generator = DataGenerator(cfg, 'test')

    # U-Net model
    model = unet(cfg)

    # Adam optimizer
    optim = tf.compat.v1.train.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
        name='Adam'
    )

    model.compile(
        optimizer='adam',
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train model on dataset
    model.fit(
        x=training_generator,
        epochs=cfg.TRAINER.EPOCHS,
        use_multiprocessing=False,
        workers=cfg.DATALOADER.NUM_WORKER,
        verbose=True,
        callbacks=[TestCallback()]
    )

    model.evaluate(
        x=validation_generator,
        use_multiprocessing=False,
        workers=cfg.DATALOADER.NUM_WORKER,
        verbose=True
    )