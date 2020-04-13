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


def SpectralNormalization(layers.Layer):
    def __init__(self,
                 n_power_iters=1,
                 initializer=initializers.RandomNomral(),
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
                                 initializers=self.initializer,
                                 trainable=False)

    def call(self, inputs, training=None):
        w_shape = inputs.get_shape() # Conv2D case: (k, k, ch1, ch2)
        w = tf.reshpae(inputs, [-1, w_shape[-1]]) # (kxkxch1, ch2)
        
        u = self.u # (kxkxch1, 1)
        for _ in range(num_iters):
            v = tf.nn.l2_normalize(tf.matmul(w, u, transpose_a=True))  # (ch2, 1)
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
                 sn_iters=1,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNomral(),
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        assert len(kernel_size) == 2, 'kernel_size must be 2 length'
        self.kernel_size = kernel_size
        assert len(strides) == 2, 'strides must be 2 length'
        self.strides = strides
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
        x = tf.nn.bias_add(x, self.bias)
        return x


def SNConv1x1(filters,
              sn_iters=1,
              kernel_initializer=initializers.Orthogonal(),
              bias_initializer=initializers.Zeros(),
              u_initializer=initializers.RandomNomral(),
              **kwargs):
    return SNConv2D(filters=filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    sn_iters=sn_iters,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    u_initializer=u_initializer,
                    **kwargs)


class SNLinear(layers.Layer):
    def __init__(self,
                 units,
                 sn_iters=1,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNomral(),
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
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
        self.bias = self.add_weight('bias',
                                    shape=[self.units],
                                    initializer=self.bias_initializer)
        self.sn = SpectralNormalization(n_power_iters=self.sn_iters,
                                        initializer=self.u_initializer,
                                        name='spectral_normalization')

    def call(self, inputs, training=None):
        kernel = self.sn(self.kernel, training)
        x = tf.matmul(inputs, kernel) + self.bias
        return x


class SNEmbedding(layers.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 sn_iters=1,
                 embeddings_initializer=initializers.RandomUniform(),
                 u_initializer=initializers.RandomNomral(),
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sn_iters = sn_iters
        self.embeddings_initializer = embeddings_initializer
        self.u_initializer = u_initializer

    def build(self, input_shape):
        self.embeddings = self.add_weight('embeddings',
                                          shape=[self.input_dim, self.output_dim],
                                          initializer=self.embeddings_initializer)
        self.sn = SpectralNormalization(n_power_iters=self.sn_iters,
                                        initializer=self.u_initializer,
                                        name='spectral_normalization')

    def call(self, inputs, training=None):
        embeddings_T = self.sn(tf.transpose(self.embeddings), training)
        embeddings = tf.transpose(embeddings_T)
        x = tf.nn.embedding_lookup(embed_map_bar, inputs)
        return x


class SelfAttention(layers.Layer):
    def __init__(self,
                 initializer=tf.initializers.orthogonal(),
                 name='self_attention'):
        super(SelfAttention, self).__init__(name=name)
        self.initializer = initializer

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.conv_theta = SNConv1x1(in_channels // 8,
                                    initializer=self.initializer,
                                    name='sn_conv_theta')
        self.conv_phi = SNConv1x1(in_channels // 8,
                                  initializer=self.initializer,
                                  name='sn_conv_phi')
        self.conv_g = SNConv1x1(in_channels // 2,
                                initializer=self.initializer,
                                name='sn_conv_g')
        self.conv_attn = SNConv1x1(in_channels,
                                   initializer=self.initializer,
                                   name='sn_conv_attn')
        self.sigma = self.add_weight('sigma',
                                     shape=[],
                                     initializer=tf.zeros_initializer())

    def call(self, x, training=None):
        batch_size, h, w, in_channels = map(int, x.shape.as_list())
        location_num = h * w
        downsampled_num = location_num // 4

        theta = self.conv_theta(x, training=training)
        theta = tf.reshape(theta, [batch_size, location_num, in_channels // 8])

        phi = self.conv_phi(x, training=training)
        phi = tf.nn.max_pool(phi, ksize=[2, 2], strides=2, padding='VALID')
        phi = tf.reshape(phi, [batch_size, downsampled_num, in_channels // 8])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(x, training=training)
        g = tf.nn.max_pool(g, ksize=[2, 2], strides=2, padding='VALID')
        g = tf.reshape(g, [batch_size, downsampled_num, in_channels // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, in_channels // 2])
        attn_g = self.conv_attn(attn_g, training=training)

        return x + self.sigma * attn_g


class ConditionalBatchNorm(layers.Layer):
    def __init__(self,
                 num_features,
                 axis=-1,
                 momentum=0.999,
                 epsilon=1E-5,
                 initializer=tf.initializers.orthogonal(),
                 name='conditional_batch_norm'):
        super(ConditionalBatchNorm, self).__init__(name=name)
        self.num_features = num_features
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.bn = tf.keras.layers.BatchNormalization(axis,
                                                     momentum,
                                                     epsilon,
                                                     center=False,
                                                     scale=False,
                                                     name='bn')
        self.linear_beta = SNLinear(num_features,
                                    use_bias=False,
                                    initializer=initializer,
                                    name='sn_linear_beta')
        self.linear_gamma = SNLinear(num_features,
                                     use_bias=False,
                                     initializer=initializer,
                                     name='sn_linear_gamma')

    def call(self, inputs, training=None):
        x, condition = inputs
        beta = self.linear_beta(condition, training=training)
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
        gamma = self.linear_gamma(condition, training=training)
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)

        x = self.bn(x, training=training)
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

    sn = spectral_norm
    conv2d = SNConv2d(filters)
    conv1x1 = SNConv1x1(filters)
    linear = SNLinear(units)
    embed0 = layers.Embedding(num_classes, embedding_size)
    embed = SNEmbedding(num_classes, embedding_size)
    self_attn = SelfAttention()
    cbn = ConditionalBatchNorm(input_dim)

    # Create sample inputs
    weights = tf.random.normal((*kernel_size, input_dim, output_dim),
                               dtype=tf.float32)
    weights = tf.Variable(weights)
    u = tf.random.normal((1, output_dim), dtype=tf.float32)
    u = tf.Variable(u)
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
    u_np, s_np, vt_np = sp.linalg.svd(w_np.T)
    u0_np = u_np[:, 0]
    for _ in range(100):
        _ = sn(weights, u, num_iters, training)
    u_pseudo = u.numpy()
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
        _ = sn(weights, u, num_iters, training)
    u_fix = u.numpy()
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
