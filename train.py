import os
import sys
import time
import tempfile
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from functools import partial

from biggan.models import Generator, Discriminator, BigGAN64
from biggan.losses import d_hinge_loss, g_hinge_loss
from biggan.callbacks import SaveGeneratedImages


def preprocess(image, label, image_size):
    image = tf.cast(image, dtype=tf.float32) / 127.5 - 1
    label = tf.cast(label, dtype=tf.int32)
    image = tf.image.resize(image, image_size)
    return image, label


def train(config, logdir):
    epochs = 10
    batch_size = 1
    z_dim = 100
    image_size = (64, 64)
    num_classes = 2
    base_dim = 64
    embedding_size = 128

    ds = tfds.load('cats_vs_dogs', split='train', as_supervised=True)
    num_examples = 23262
    steps_per_epoch = num_examples // batch_size
    ds = ds.shuffle(20000)\
      .map(partial(preprocess, image_size=image_size),
           num_parallel_calls=tf.data.experimental.AUTOTUNE)\
      .batch(batch_size)\
      .repeat(epochs)\
      .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # create model
    generator = Generator(num_classes,
                          base_dim,
                          embedding_size,
                          name='generator')
    discriminator = Discriminator(num_classes, base_dim, name='discriminator')
    biggan = BigGAN64(z_dim,
                      num_classes,
                      generator,
                      discriminator,
                      name='biggan')

    biggan.compile(g_optimizer=tf.keras.optimizers.Adam(),
                   d_optimizer=tf.keras.optimizers.Adam(),
                   g_loss_fn=g_hinge_loss,
                   d_loss_fn=d_hinge_loss)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        SaveGeneratedImages(num_examples=9, logdir=logdir + '/images')
    ]
    biggan.fit(ds,
               batch_size=batch_size,
               epochs=epochs,
               steps_per_epoch=steps_per_epoch,
               callbacks=callbacks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.debug:
            logdir = tmpdir
        else:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            logdir = f'logs/{datetime.now().strftime("%Y-%m-%dT%H:%M")}'
        train(args, logdir)


if __name__ == '__main__':
    main()
