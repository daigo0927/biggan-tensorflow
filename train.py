import os
import sys
import time
import tempfile
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from biggan.datasets import Cat
from biggan.models import Generator, Discriminator
from biggan.losses import discriminator_hinge_loss,\
     discriminator_bce_loss,\
     generator_hinge_loss, \
     generator_bce_loss
from biggan.utils import make_z_normal, make_label_uniform, prepare_parser, save_args


def main():
    parser = prepare_parser()
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


def train(config, logdir):
    # ---------------- Input definition ----------------
    datagen = ImageDataGenerator(**config['data']['transform'])
    datagen = datagen.flow_from_directory(# TODO)

    # ---------------- Models and optimizers Building ----------------
    discriminator = Discriminator(num_classes=dataset.num_classes,
                                  df_dim=args.df_dim,
                                  name='discriminator')
    generator = Generator(num_classes=dataset.num_classes,
                          gf_dim=args.gf_dim,
                          embedding_size=args.embedding_size,
                          name='generator')

    d_opt = tf.keras.optimizers.Adam(args.discriminator_learning_rate,
                                     args.beta1, args.beta2)
    g_opt = tf.keras.optimizers.Adam(args.generator_learning_rate,
                                     args.beta1, args.beta2)
    
    # -------------- Define update functions ----------------
    @tf.function
    def update_d(images, labels):
        zs = make_z_normal(args.batch_size, args.z_dim)
        labels_fake = make_label_uniform(args.batch_size, dataset.num_classes)
        with tf.GradientTape() as tape:
            images_fake = generator((zs, labels_fake), training=False)
            logits_real = discriminator((images, labels), training=True)
            logits_fake = discriminator((images_fake, labels_fake), training=True)
            loss_real = get_d_real_loss(logits_real)
            loss_fake = get_d_fake_loss(logits_fake)
            loss = loss_real + loss_fake
        grads = tape.gradient(loss, discriminator.trainable_weights)
        d_opt.apply_gradients(zip(grads, discriminator.trainable_weights))
        out = {'loss': loss,
               'loss_real': loss_real,
               'loss_fake': loss_fake}
        return out

    @tf.function
    def update_g():
        zs = make_z_normal(args.batch_size, args.z_dim)
        labels_fake = make_label_uniform(args.batch_size, dataset.num_classes)
        with tf.GradientTape() as tape:
            images_fake = generator((zs, labels_fake), training=True)
            logits_fake = discriminator((images_fake, labels_fake), training=False)
            loss = get_g_loss(logits_fake)
        grads = tape.gradient(loss, generator.trainable_weights)
        g_opt.apply_gradients(zip(grads, generator.trainable_weights))
        images_fake_vis = tf.cast((images_fake+1.0)*127.5, dtype=tf.uint8)
        out = {'images_fake': images_fake_vis,
               'loss': loss}
        return out

    # ---------------- Log preparation ----------------
    summary_writer = tf.summary.create_file_writer(logdir)
    save_args(args, logdir+'/args.json')
    print(f'Graph successfully built. Histories are logged in {logdir}')
    print(f'run \'$tensorboard --logdir={logdir}\' to see the training logs.')

    # ------------- Actual training iteration ---------------
    dataiter = iter(dataset.loader)
    for i in tqdm(range(args.num_iters), desc='Trainig'):
        images, labels = dataiter.get_next()
        d_out = update_d(images, labels)
        if i%args.n_discriminator_update == 0:
            g_out = update_g()

        if i == 0 or (i+1)%args.check_step == 0:
            generator.save_weights(logdir+'/generator.ckpt')
            discriminator.save_weights(logdir+'/discriminator.ckpt')
            with summary_writer.as_default():
                tf.summary.scalar('d/loss', d_out['loss'], step=i+1)
                tf.summary.scalar('d/loss_real', d_out['loss_real'], step=i+1)
                tf.summary.scalar('d/loss_fake', d_out['loss_fake'], step=i+1)
                tf.summary.scalar('g/loss', g_out['loss'], step=i+1)
                tf.summary.image('generated_images', g_out['images_fake'],
                                 step=i+1, max_outputs=args.num_visualize)
                summary_writer.flush()


if __name__ == '__main__':
    main()
