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
    prep_fn = lambda image: (image/127.5)-1.0
    datagen = ImageDataGenerator(preprocessing_function=prep_fn,
                                 **data_config['transform'])

    # ---------------- Models and optimizers Building ----------------
    discriminator = DCDiscriminator(**dconfig)
    generator = DCGenerator(**gconfig)

    d_opt = tf.keras.optimizers.Adam(**train_config['discriminator_optimizer'])
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
    @tf.function
    def update(images):
        z = make_z_normal(batch_size, train_config['z_dim'])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            images_fake = generator(z, training=True)
            
            logits_real = discriminator(images, training=True)
            logits_fake = discriminator(images_fake, training=True)

            d_loss = d_loss_fn(logits_real, logits_fake)
            g_loss = g_loss_fn(logits_fake)

        d_grad = d_tape.gradient(d_loss, discriminator.trainable_weights)
        g_grad = g_tape.gradient(g_loss, generator.trainable_weights)

        d_opt.apply_gradients(zip(d_grad, discriminator.trainable_weights))
        g_opt.apply_gradients(zip(g_grad, generator.trainable_weights))

        return d_loss, g_loss, images_fake

    # Create and define summary protocol
    writer = tf.summary.create_file_writer(logdir)
    def summarize(d_loss, g_loss, images_fake, step):
        with writer.as_default():
            tf.summary.scalar('loss/discriminator', d_loss, step=step)
            tf.summary.scalar('loss/generator', g_loss, step=step)
            images_fake_vis = tf.cast(127.5*(images_fake+1.0), tf.uint8)
            tf.summary.image('sample', images_fake_vis, step=step)
        writer.flush()

    # Create checkpoint writer
    ckptdir = logdir+'/checkpoints'
    ckpt_prefix = ckptdir+'/ckpt'
    checkpoint = tf.train.Checkpoint(discriminator_optimzier=d_opt,
                                     generator_optimizer=g_opt,
                                     discriminator=discriminator,
                                     generator=generator)

    # ------------- Actual training iteration ---------------
    global_step = 1
    for e in range(train_config['epochs']):
        loader = datagen.flow_from_directory(**data_config['flow'])
        for i in tqdm(range(len(loader)), desc=f'Epoch {e}'):
            images, labels = loader.next()
            d_loss, g_loss, images_fake = update(images)

            if global_step == 1 or global_step%train_config['record_step']:
                summarize(d_loss, g_loss, images_fake, global_step)

            global_step += 1
                    
        checkpoint.save(file_prefix=ckpt_prefix)


def main():
    parser = ArgumentParser(description='GAN training parser')
    parser.add_argument('config', type=str,
                        help='Experiment config file')
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
            logd = config['train']['output_dir']
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
