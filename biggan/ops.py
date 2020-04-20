''' Implementations of GAN modules '''

import tensorflow as tf
from tensorflow.keras import layers, initializers

# COPYRIGHT for implementation of spectral normalization.
#
# MIT License

# Copyright (c) 2019 Eon Kim

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Additional reference: https://github.com/chainer/chainer/blob/v7.1.0/chainer/link_hooks/spectral_normalization.py#L79


class SpectralNormalization(layers.Wrapper):
    def __init__(self, layer, n_power_iters=1, weight_name='kernel', **kwargs):
        if not isinstance(layer, layers.Layer):
            raise ValueError(f'Invalid layer argument was given: {layer}')
        super().__init__(layer, **kwargs)
        self.n_power_iters = n_power_iters
        self.weight_name = weight_name
        self.original_weight = None

    def build(self, input_shape):
        self.layer.build(input_shape)
        weight = getattr(self.layer, self.weight_name)
        self.w_shape = weight.shape.as_list()
        self.u = self.add_weight(
            'u',
            shape=[self.w_shape[-1], 1],
            dtype=tf.float32,
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False)

        super().build()

    def call(self, inputs, training=None):
        self.update_weight()
        outputs = self.layer(inputs)
        self.restore_weight()
        return outputs

    def update_weight(self):
        self.original_weight = getattr(self.layer, self.weight_name)
        weight = self.original_weight

        w = tf.reshape(weight, [-1, self.w_shape[-1]])

        u = self.u
        for _ in range(self.n_power_iters):
            v = tf.nn.l2_normalize(tf.matmul(w, u))  # (kxkxch1, 1)
            u = tf.nn.l2_normalize(tf.matmul(w, v,
                                             transpose_a=True))  # (ch2, 1)

        sigma = tf.matmul(tf.matmul(v, w, transpose_a=True), u)  # (1, 1)
        self.u.assign(u)
        setattr(self.layer, self.weight_name, weight / sigma)

    def restore_weight(self):
        setattr(self.layer, self.weight_name, self.original_weight)


def SNConv2D(filters, kernel_size, strides=(1, 1), **kwargs):
    return SpectralNormalization(layers.Conv2D(filters, kernel_size, strides,
                                               **kwargs),
                                 n_power_iters=1,
                                 name='sn_conv')


def SNConv1x1(filters, **kwargs):
    return SNConv2D(filters=filters, kernel_size=(1, 1), **kwargs)


def SNLinear(units, **kwargs):
    return SpectralNormalization(layers.Dense(units, **kwargs),
                                 n_power_iters=1,
                                 name='sn_linear')


class TransposedEmbedding(layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            'embeddings',
            shape=[self.output_dim, self.input_dim],
            dtype=tf.float32)

    def call(self, inputs):
        embeddings = tf.transpose(self.embeddings)
        outputs = tf.nn.embedding_lookup(embeddings, inputs)
        return outputs


def SNEmbedding(input_dim, output_dim, **kwargs):
    return SpectralNormalization(TransposedEmbedding(input_dim, output_dim,
                                                     **kwargs),
                                 n_power_iters=1,
                                 weight_name='embeddings',
                                 name='sn_embedding')


class SNSelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv_theta = SNConv1x1(in_channels // 8,
                                    use_bias=False,
                                    name='sn_conv_theta')
        self.conv_phi = SNConv1x1(in_channels // 8,
                                  use_bias=False,
                                  name='sn_conv_phi')
        self.conv_g = SNConv1x1(in_channels // 2,
                                use_bias=False,
                                name='sn_conv_g')
        self.conv_attn = SNConv1x1(in_channels,
                                   use_bias=False,
                                   name='sn_conv_attn')
        self.sigma = self.add_weight('sigma',
                                     shape=[],
                                     initializer=tf.zeros_initializer())

    def call(self, inputs, training=None):
        _, h, w, in_channels = inputs.shape.as_list()
        location_num = h * w
        downsampled_num = location_num // 4

        theta = self.conv_theta(inputs, training)
        theta = tf.reshape(theta, [-1, location_num, in_channels // 8])

        phi = self.conv_phi(inputs, training)
        phi = tf.nn.max_pool(phi, ksize=[2, 2], strides=2, padding='VALID')
        phi = tf.reshape(phi, [-1, downsampled_num, in_channels // 8])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(inputs, training)
        g = tf.nn.max_pool(g, ksize=[2, 2], strides=2, padding='VALID')
        g = tf.reshape(g, [-1, downsampled_num, in_channels // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [-1, h, w, in_channels // 2])
        attn_g = self.conv_attn(attn_g, training)

        return inputs + self.sigma * attn_g


class ConditionalBatchNorm(layers.Layer):
    def __init__(self, axis=-1, momentum=0.1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        x_shape, condition_shape = input_shape
        x_channels = x_shape[-1]

        self.bn = layers.BatchNormalization(self.axis,
                                            self.momentum,
                                            self.epsilon,
                                            center=False,
                                            scale=False,
                                            name='bn')
        self.linear_beta = SNLinear(x_channels,
                                    use_bias=False,
                                    name='sn_linear_beta')
        self.linear_gamma = SNLinear(x_channels,
                                     use_bias=False,
                                     name='sn_linear_gamma')

    def call(self, inputs, training=None):
        x, condition = inputs
        beta = self.linear_beta(condition, training)
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
        gamma = self.linear_gamma(condition, training)
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)

        x = self.bn(x, training)
        x = (1.0 + gamma) * x + beta
        return x


if __name__ == '__main__':
    # Settings
    batch_size = 4
    filters = 32
    kernel_size = (3, 3)
    strides = (2, 2)
    units = 128
    num_classes = 10
    embedding_size = 128
    num_iters = 1

    # Create layer instances
    conv2d = SNConv2D(filters, kernel_size, strides)
    conv1x1 = SNConv1x1(filters)
    linear = SNLinear(units)
    embed = SNEmbedding(num_classes, embedding_size)
    self_attn = SNSelfAttention()
    cbn = ConditionalBatchNorm()

    # Create sample inputs
    images = tf.random.normal((batch_size, 32, 32, 3), dtype=tf.float32)
    features = tf.random.normal((batch_size, 100), dtype=tf.float32)
    labels = tf.random.uniform((batch_size, ),
                               minval=0,
                               maxval=10,
                               dtype=tf.dtypes.int32)

    # Forward into layers
    training = True
    _ = conv2d(images, training=training)
    _ = conv1x1(images, training=training)
    _ = linear(features, training=training)
    _ = embed(labels, training=training)
    _ = self_attn(images, training=training)
    _ = cbn([images, features], training=training)

    # Singular value estimation of spectral normalization
    import numpy as np
    norm = np.linalg.norm
    import scipy as sp
    kernel = conv2d.layer.kernel
    w_np = kernel.numpy().reshape((-1, filters))
    u_np, s_np, vt_np = sp.linalg.svd(w_np.T)
    u0_np = u_np[:, 0]
    for _ in range(100):
        _ = conv2d(images, training)
    u_pseudo = conv2d.u.numpy()[:, 0]
    cossim = np.abs(np.sum(u0_np * u_pseudo)) / norm(u0_np) / norm(u_pseudo)
    if cossim > 0.95:
        print(f'Singular vector similarity: {cossim}')
    else:
        raise ValueError(
            f'Singular vector estimation failed, cosine similarity: {cossim}')

    training = False
    for _ in range(100):
        _ = conv2d(images, training)
    u_fix = conv2d.u.numpy()[:, 0]
    cossim = np.abs(np.sum(u_pseudo * u_fix)) / norm(u_pseudo) / norm(u_fix)
    if cossim > 0.95:
        print(f'Singular vector preservation : {cossim}')
    else:
        raise ValueError(
            f'Singular vector preservation failed, cosine similarity: {cossim}'
        )
    _ = conv2d(images, training=training)
    _ = conv1x1(images, training=training)
    _ = linear(features, training=training)
    _ = embed(labels, training=training)
    _ = self_attn(images, training=training)
    _ = cbn([images, features], training=training)

    print('Completed.')
