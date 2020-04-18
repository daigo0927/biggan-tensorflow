import os
import imageio
import numpy as np
import tensorflow as tf


def batch_to_grid(images):
    batch_size, h, w, ch = images.shape
    n_rows = int(np.sqrt(batch_size))
    n_cols = n_rows if n_rows**2 == batch_size else n_rows + 1

    grid_images = np.zeros((n_rows * h, n_cols * w, ch), dtype='int32')
    for i, image in enumerate(images):
        i_row, i_col = i // n_cols, i % n_cols
        grid_images[i_row * h:(i_row + 1) * h,
                    i_col * w:(i_col + 1) * w] = image
    return grid_images


class SaveGeneratedImages(tf.keras.callbacks.Callback):
    def __init__(self, num_examples=1, logdir='./images'):
        super().__init__()
        self.num_examples = num_examples
        self.logdir = logdir

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.zs = None
        self.labels_fake = None

    def on_epoch_end(self, epoch, logs=None):
        if self.zs is None and self.labels_fake is None:
            latent_dim = self.model.latent_dim
            num_classes = self.model.num_classes

            self.zs = tf.random.normal((self.num_examples, latent_dim))
            self.labels_fake = tf.random.uniform((self.num_examples, ),
                                                 0,
                                                 num_classes,
                                                 dtype=tf.int32)

        images_fake = self.model.generator([self.zs, self.labels_fake],
                                           training=False)
        images_fake = images_fake.numpy()
        images_fake = ((images_fake + 1) * 127.5).astype('uint8')
        images_fake = batch_to_grid(images_fake)

        filename = self.logdir + f'/sample_{epoch+1}epoch.png'
        imageio.imwrite(filename, images_fake)
