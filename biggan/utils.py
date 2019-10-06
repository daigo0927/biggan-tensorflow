import json
import tensorflow as tf
from argparse import ArgumentParser
from collections import OrderedDict


def make_z_normal(batch_size, z_dim):
    shape = [batch_size, z_dim]
    z = tf.random.normal(shape, name='z', dtype=tf.float32)
    return z


def make_label_uniform(batch_size, num_classes):
    shape = [batch_size]
    labels = tf.random.uniform(shape, 0, num_classes, tf.int32)
    return labels


def prepare_parser():
    parser = ArgumentParser(description='BigGAN training configs')
    # Dataset config
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Target dataset class name. (required)')
    parser.add_argument('-dd', '--dataset_dir', type=str, required=True,
                        help='Target dataset directory. (required)')
    # Iteration config
    parser.add_argument('-i', '--num_iters', type=int, default=100,
                        help='Number of all iteration. [100]')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size. [4]')
    # Data preprocessing config
    parser.add_argument('--resize_shape', nargs=2, type=int, default=None,
                        help='Resize shape [None]')
    parser.add_argument('--crop_shape', nargs=2, type=int, default=None,
                        help='Crop shape for images. [None]')
    parser.add_argument('--rotate', action='store_true',
                        help='Enable rotation in preprocessing')
    parser.add_argument('--flip_lr', action='store_true',
                        help='Enable left-right flip in preprocessing')
    parser.add_argument('--flip_ud', action='store_true',
                        help='Enable up-down flip in preprocessing')
    # Model and optimization config
    parser.add_argument('--z_dim', type=int, default=120,
                        help='Dimensionality of latent code z. [120]')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Dimensionality of class embedding. [128]')
    parser.add_argument('--gf_dim', type=int, default=64,
                        help='Dimensionality of generator filter. [64]')
    parser.add_argument('--df_dim', type=int, default=64,
                        help='Dimensionality of discriminator filter. [64]')
    parser.add_argument('-glr', '--generator_learning_rate',
                        type=float, default = 0.00005,
                        help='Learning rate of generator for Adam [0.00005]')
    parser.add_argument('-dlr', '--discriminator_learning_rate',
                        type=float, default=0.0002,
                        help='Learning rate of discriminator for Adam [0.0002]')
    parser.add_argument('-nd', '--n_discriminator_update', type=int, default=2,
                        help='Relative update step of discriminator to G [2]')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Beta1 term of Adam [0.0]')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 term of Adam [0.999]')
    parser.add_argument('-nv', '--num_visualize', type=int, default=4,
                        help='Number of figures to be saved. [4]')
    parser.add_argument('-r', '--resume', type = str, default = None,
                        help = 'Learned parameter checkpoint file. (optional)')
    parser.add_argument('--debug', action='store_true',
                        help='Validate graph construction, stop before training.')
    return parser


def save_args(args, filename):
    args = OrderedDict(vars(args))
    with open(filename, 'w') as f:
        json.dump(args, f, indent=4)
