import os
import sys
import time
import yaml
import tempfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from biggan.datasets import Cat
from biggan.models import DCGenerator, DCDiscriminator
from biggan.losses import d_hinge_loss, d_bce_loss, g_hinge_loss, g_bce_loss
from biggan.utils import make_z_normal, make_label_uniform


def train(config, logdir):
    try:
        data_config = config['data']
        batch_size = data_config['flow']['batch_size']

        model_config = config['model']
        darch = model_config['discriminator_architecture']
        dconfig = model_config['discriminator_config']
        garch = model_config['generator_architecture']
        gconfig = model_config['generator_config']

        train_config = config['train']
    except KeyError:
        sys.exit('Invalid key detected')
        
    # ---------------- Input definition ----------------
    datagen = ImageDataGenerator(**data_config['transform'])
    datagen = datagen.flow_from_directory(**data_config['flow'])

    # ---------------- Models and optimizers Building ----------------
    discriminator = DCDiscriminator(**dconfig)
    generator = DCGenerator(**gconfig)

    d_opt = tf.keras.optimizers.Adam(**train_config['discriminator_optimzier'])
    g_opt = tf.keras.optimizers.Adam(**train_config['generator_optimizer'])

    # ---------------- Loss function definition ----------------
    if train_config['loss'] == 'bce':
        g_loss_fn = g_bce_loss
        d_loss_fn = d_bce_loss
    elif train_config['loss'] == 'hinge':
        g_loss_fn = g_hinge_loss
        d_loss_fn = d_hinge_loss
    else:
        raise KeyError('invalid loss function detected')
    
    # -------------- Define update functions ----------------
    # @tf.function
    # def update_d(images, labels):
    #     zs = make_z_normal(args.batch_size, args.z_dim)
    #     labels_fake = make_label_uniform(args.batch_size, dataset.num_classes)
    #     with tf.GradientTape() as tape:
    #         images_fake = generator((zs, labels_fake), training=False)
    #         logits_real = discriminator((images, labels), training=True)
    #         logits_fake = discriminator((images_fake, labels_fake), training=True)
    #         loss_real = get_d_real_loss(logits_real)
    #         loss_fake = get_d_fake_loss(logits_fake)
    #         loss = loss_real + loss_fake
    #     grads = tape.gradient(loss, discriminator.trainable_weights)
    #     d_opt.apply_gradients(zip(grads, discriminator.trainable_weights))
    #     out = {'loss': loss,
    #            'loss_real': loss_real,
    #            'loss_fake': loss_fake}
    #     return out

    # @tf.function
    # def update_g():
    #     zs = make_z_normal(args.batch_size, args.z_dim)
    #     labels_fake = make_label_uniform(args.batch_size, dataset.num_classes)
    #     with tf.GradientTape() as tape:
    #         images_fake = generator((zs, labels_fake), training=True)
    #         logits_fake = discriminator((images_fake, labels_fake), training=False)
    #         loss = get_g_loss(logits_fake)
    #     grads = tape.gradient(loss, generator.trainable_weights)
    #     g_opt.apply_gradients(zip(grads, generator.trainable_weights))
    #     images_fake_vis = tf.cast((images_fake+1.0)*127.5, dtype=tf.uint8)
    #     out = {'images_fake': images_fake_vis,
    #            'loss': loss}
    #     return out

    def summarize(images_fake, d_loss, g_loss, step):
        tf.summary.scalar('loss/discriminator', d_loss, step=step)
        tf.summary.scalar('loss/generator', g_loss, step=step)

    @tf.function
    def update(image, step):
        z = make_z_normal(batch_size, train_config['z_dim'])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            images_fake = generator(z, training=True)
            images_fake_vis = tf.cast(127.5*(images_fake+1.0), tf.uint8)
            
            logits_real = discriminator(images, training=True)
            logits_fake = discriminator(images_fake, training=True)

            d_loss = d_loss_fn(logits_real, logits_fake)
            g_loss = g_loss_fn(logits_fake)

        d_grad = d_tape.geadient(d_loss, discriminator.trainable_weights)
        g_grad = g_tape.geadient(g_loss, generator.trainable_weights)

        d_opt.apply_gradients(zip(d_grad, discriminator.trainable_weights))
        g_opt.apply_gradients(zip(g_grad, generator.trainable_weights))

        with writer.as_default():
            if tf.cond(step%100 == 0,
                       summarize(images_fake_vis, d_loss, g_loss, step),
                       lambda: 0)


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


def main():
    parser = ArgumentParser(description='GAN training parser')
    parser.add_argument('config', type=str, required=True,
                        help='Experiment config file (required)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Run on debug mode, remove created files automatically')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.debug:
            logdir = tmpdir
            config['train']['epochs'] = 1
        else:
            logd = config['output_dir']
            logdir = os.path.join(logd, datetime.now().strftime("%Y-%m-%dT%H-%M"))
            if not os.path.exists(logdir):
                os.makedirs(logdir)

        print('---------------------------')
        print('Log directory: {}'.format(logdir))
        print('---------------------------')

        config_name = args.config.split('/')[-1]
        savename = '/'.join([logdir, config_name])
        with open(savename, 'w') as f:
            f.write(yaml.dump(config, default_flow_style=False))

        train(config, logdir)
        

if __name__ == '__main__':
    main()
