import tensorflow as tf
from tensorflow.keras import layers, initializers

# COPYRIGHT for implementation of spectral normalization.
# coding=utf-8
# Copyright 2020 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class SpectralNormalization(layers.Layer):
    def __init__(self,
                 n_power_iters=1,
                 initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.n_power_iters = n_power_iters
        self.initializer = initializer

    def build(self, input_shape):
        # Dummy weights for shape inference
        w = tf.reshape(tf.zeros(input_shape), [-1, input_shape[-1]])
        self.u = self.add_weight('u',
                                 shape=[w.shape[0], 1],
                                 dtype=tf.float32,
                                 initializer=self.initializer,
                                 trainable=False)

    def call(self, inputs, training=None):
        w_shape = inputs.get_shape()  # Conv2D case: (k, k, ch1, ch2)
        w = tf.reshape(inputs, [-1, w_shape[-1]])  # (kxkxch1, ch2)

        u = self.u  # (kxkxch1, 1)
        for _ in range(num_iters):
            v = tf.nn.l2_normalize(tf.matmul(w, u,
                                             transpose_a=True))  # (ch2, 1)
            u = tf.nn.l2_normalize(tf.matmul(w, v))  # (kxkxch1, 1)

        if training:
            self.u.assign(u)
            u = tf.identity(u)

        u = tf.stop_gradient(u)
        v = tf.stop_gradient(v)

        # Spectral norm
        norm = tf.matmul(tf.matmul(u, w, transpose_a=True), v)  # (1, 1)
        # Normalization
        w_normalized = w / norm
        return tf.reshape(w_normalized, w_shape)


class SNConv2D(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 use_bias=True,
                 sn_iters=1,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        assert len(kernel_size) == 2, 'kernel_size must be 2 length'
        self.kernel_size = kernel_size
        assert len(strides) == 2, 'strides must be 2 length'
        self.strides = strides
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.u_initializer = u_initializer

    def build(self, input_shape):
        in_channels = input_shape[-1]
        kernel_shape = [*self.kernel_size, in_channels, self.filters]

        self.kernel = self.add_weight('kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.filters],
                                        initializer=self.bias_initializer)
        self.sn = SpectralNormalization(n_power_iters=self.sn_iters,
                                        initializer=self.u_initializer,
                                        name='spectral_normalization')

    def call(self, inputs, training=None):
        kernel = self.sn(self.kernel, training)
        x = tf.nn.conv2d(inputs,
                         kernel,
                         strides=[1, *self.strides, 1],
                         padding='SAME')
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


def SNConv1x1(filters,
              use_bias=True,
              sn_iters=1,
              kernel_initializer=initializers.Orthogonal(),
              bias_initializer=initializers.Zeros(),
              u_initializer=initializers.RandomNormal(),
              **kwargs):
    return SNConv2D(filters=filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    use_bias=use_bias,
                    sn_iters=sn_iters,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    u_initializer=u_initializer,
                    **kwargs)


class SNLinear(layers.Layer):
    def __init__(self,
                 units,
                 use_bias=True,
                 sn_iters=1,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.u_initializer = u_initializer

    def build(self, input_shape):
        in_features = input_shape[-1]
        kernel_shape = [in_features, self.units]
        self.kernel = self.add_weight('kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units],
                                        initializer=self.bias_initializer)
        self.sn = SpectralNormalization(n_power_iters=self.sn_iters,
                                        initializer=self.u_initializer,
                                        name='spectral_normalization')

    def call(self, inputs, training=None):
        kernel = self.sn(self.kernel, training)
        x = tf.matmul(inputs, kernel)
        if self.use_bias:
            x += self.bias
        return x


class SNEmbedding(layers.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 sn_iters=1,
                 embeddings_initializer=initializers.RandomUniform(),
                 u_initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sn_iters = sn_iters
        self.embeddings_initializer = embeddings_initializer
        self.u_initializer = u_initializer

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            'embeddings',
            shape=[self.input_dim, self.output_dim],
            initializer=self.embeddings_initializer)
        self.sn = SpectralNormalization(n_power_iters=self.sn_iters,
                                        initializer=self.u_initializer,
                                        name='spectral_normalization')

    def call(self, inputs, training=None):
        embeddings_T = self.sn(tf.transpose(self.embeddings), training)
        embeddings = tf.transpose(embeddings_T)
        x = tf.nn.embedding_lookup(embeddings, inputs)
        return x


class SNSelfAttention(layers.Layer):
    def __init__(self,
                 use_bias=False,
                 sn_iters=1,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.u_initializer = u_initializer

        self._params = {
            'use_bias': use_bias,
            'sn_iters': sn_iters,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'u_initializer': u_initializer
        }

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.conv_theta = SNConv1x1(in_channels // 8,
                                    name='sn_conv_theta',
                                    **self._params)
        self.conv_phi = SNConv1x1(in_channels // 8,
                                  name='sn_conv_phi',
                                  **self._params)
        self.conv_g = SNConv1x1(in_channels // 2,
                                name='sn_conv_g',
                                **self._params)
        self.conv_attn = SNConv1x1(in_channels,
                                   name='sn_conv_attn',
                                   **self._params)
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
    def __init__(self,
                 axis=-1,
                 momentum=0.999,
                 epsilon=1E-5,
                 use_bias=False,
                 sn_iters=1,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.u_initializer = u_initializer

        self._params = {
            'use_bias': use_bias,
            'sn_iters': sn_iters,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'u_initializer': u_initializer
        }

        # TODO: implement use_bias argment, referred as https://github.com/huggingface/pytorch-pretrained-BigGAN/blob/master/pytorch_pretrained_biggan/model.py
        # official TF implementation: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/layers/convolutional.py#L48

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
                                    name='sn_linear_beta',
                                    **self._params)
        self.linear_gamma = SNLinear(x_channels,
                                     name='sn_linear_gamma',
                                     **self._params)

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
    # Create layer instances
    batch_size = 4
    kernel_size = (3, 3)
    filters = 32
    units = 128
    num_classes = 10
    input_dim = 3
    output_dim = 128
    embedding_size = 128
    num_iters = 1

    sn = SpectralNormalization()
    conv2d = SNConv2D(filters)
    conv1x1 = SNConv1x1(filters)
    linear = SNLinear(units)
    embed0 = layers.Embedding(num_classes, embedding_size)
    embed = SNEmbedding(num_classes, embedding_size)
    self_attn = SNSelfAttention()
    cbn = ConditionalBatchNorm(input_dim)

    # Create sample inputs
    weights = tf.random.normal((*kernel_size, input_dim, output_dim),
                               dtype=tf.float32)
    weights = tf.Variable(weights)
    images = tf.random.normal((batch_size, 64, 64, 3), dtype=tf.float32)
    features = tf.random.normal((batch_size, 100), dtype=tf.float32)
    labels = tf.random.uniform((batch_size, ),
                               minval=0,
                               maxval=10,
                               dtype=tf.dtypes.int32)

    training = True
    # Singular value estimation of spectral normalization
    import numpy as np
    norm = np.linalg.norm
    import scipy as sp
    w_np = weights.numpy().reshape((-1, output_dim))
    u_np, s_np, vt_np = sp.linalg.svd(w_np)
    u0_np = u_np[:, 0]
    for _ in range(100):
        _ = sn(weights, training)
    u_pseudo = sn.u.numpy()[:, 0]
    cossim = np.abs(np.sum(u0_np * u_pseudo)) / norm(u0_np) / norm(u_pseudo)
    if cossim > 0.95:
        print(f'Singular vector similarity: {cossim}')
    else:
        raise ValueError(
            f'Singular vector estimation failed, cosine similarity: {cossim}')
    # Forward into layers
    _ = conv2d(images, training=training)
    _ = conv1x1(images, training=training)
    _ = linear(features, training=training)
    _ = embed0(labels, training=training)
    _ = embed(labels, training=training)
    _ = self_attn(images, training=training)
    _ = cbn([images, features], training=training)

    training = False
    for _ in range(100):
        _ = sn(weights, training)
    u_fix = sn.u.numpy()[:, 0]
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
    _ = embed0(labels, training=training)
    _ = embed(labels, training=training)
    _ = self_attn(images, training=training)
    _ = cbn([images, features], training=training)

    print('Completed.')
