import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

from datasets import Cat
from models import Generator, Discriminator
from losses import get_d_real_loss, get_d_fake_loss, get_g_loss
from utils import make_z_normal, make_label_uniform, prepare_parser, save_args


def train(args):
    # --------- Dataset pipeline construction ---------------
    pipe = {'Cat': Cat}[args.dataset]
    dataset = pipe(dataset_dir=args.dataset_dir,
                   train_or_test='train',
                   batch_size=args.batch_size,
                   resize_shape=args.resize_shape,
                   crop_shape=args.crop_shape,
                   rotate=args.rotate,
                   flip_left_right=args.flip_lr,
                   flip_up_down=args.flip_ud)

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
            images_fake = generator(zs, labels_fake, training=False)
            logits_real = discriminator(images, labels, training=True)
            logits_fake = discriminator(images_fake, labels_fake, training=False)
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
            images_fake = generator(zs, labels_fake, training=True)
            logits_fake = discriminator(images_fake, labels_fake, training=False)
            loss = get_g_loss(logits_fake)
        grads = tape.gradient(loss, generator.trainable_weights)
        g_opt.apply_gradients(zip(grads, generator.trainable_weights))
        images_fake_vis = tf.cast((images_fake+1.0)*127.5, dtype=tf.uint8)
        out = {'images_fake': images_fake_vis,
               'loss': loss}
        return out

    # ---------------- Log preparation ----------------
    if not args.test:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        log_dir = f'./logs/history_{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
        summary_writer = tf.summary.create_file_writer(log_dir)
        save_args(args, log_dir+'/args.json')
        bar = tqdm(desc='Trainig loop', total=args.num_iters)

        print(f'Graph successfully built. Histories are logged in {log_dir}')
        print(f'run \'$tensorboard --logdir={log_dir}\' to see the training logs.')

    # ------------- Actual training iteration ---------------
    for i, (images, labels) in enumerate(dataset.loader):
        # Discriminator update
        d_out = update_d(images, labels)
        # Generator update
        if i%args.n_discriminator_update == 0:
            g_out = update_g()

        if args.test:
            print('--------- Running test succesfully completed ---------')
            break

        if i == 0 or (i+1)%1000 == 0:
            generator.save_weights(log_dir+'/generator.ckpt')
            discriminator.save_weights(log_dir+'/discriminator.ckpt')
            with summary_writer.as_default():
                tf.summary.scalar('d/loss', d_out['loss'], step=i+1)
                tf.summary.scalar('d/loss_real', d_out['loss_real'], step=i+1)
                tf.summary.scalar('d/loss_fake', d_out['loss_fake'], step=i+1)
                tf.summary.scalar('g/loss', d_out['loss'], step=i+1)
                tf.summary.image('generated_images', g_out['images_fake'],
                                 step=i+1, max_outputs=args.num_visualize)
                summary_writer.flush()

        if i == args.num_iters:
            print('---------- Trainig completed -------------')
            break

        bar.update(1)


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
