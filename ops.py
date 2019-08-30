import tensorflow as tf
from tensorflow.keras import layers


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5 + eps)


class SpectralNorm(layers.Layer):
    def __init__(self,
                 num_iters=1,
                 name='spectral_norm'):
        super(SpectralNorm, self).__init__(name=name)
        self.num_iters = num_iters

    def build(self, input_shape):
        # First sigular vector
        self.u = self.add_weight('u', shape=[1, int(input_shape[-1])],
                                 initializer=tf.initializers.TruncatedNormal(),
                                 trainable=False)

    def call(self, weights, training=True):
        # Because of the order of weights axis, code is a bit different from those of chainer or pytorchm,
        # but this actually holds the same process.
        # standard SVD (chainer and pytorch): W = u*Sigma*v_T
        # tensorflow: W^T = u*Sigma*v_T
        w_shape = weights.shape.as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        u_ = self.u
        for _ in range(self.num_iters):
            v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
            u_ = _l2normalize(tf.matmul(v_, w_mat))

        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
        w_mat /= sigma
        if training:
            self.u.assign(u_)
        w_bar = tf.reshape(w_mat, w_shape)
        return w_bar


class SNConv2d(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='snconv2d'):
        super(SNConv2d, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.sn_iters = sn_iters
        self.initializer = initializer

    def build(self, input_shape):
        self.spectral_norm = SpectralNorm(num_iters=self.sn_iters)
        kernel_shape = [*self.kernel_size, int(input_shape[-1]), self.filters]
        self.kernel = self.add_weight('kernel', shape=kernel_shape,
                                      initializer=self.initializer)
        self.bias = self.add_weight('bias', shape=[self.filters],
                                    initializer=tf.zeros_initializer())

    def call(self, x, training):
        w_bar = self.spectral_norm(self.kernel, training=training)
        x = tf.nn.conv2d(x, w_bar,
                         strides=[1, *self.strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.bias)
        return x


class SNConv1x1(layers.Layer):
    def __init__(self,
                 filters,
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='snconv1x1'):
        super(SNConv1x1, self).__init__(name=name)
        self.filters = filters
        self.sn_iters = sn_iters
        self.initializer = initializer

    def build(self, input_shape):
        self.spectral_norm = SpectralNorm(num_iters=self.sn_iters)
        kernel_shape = [1, 1, int(input_shape[-1]), self.filters]
        self.kernel = self.add_weight('kernel', shape=kernel_shape,
                                      initializer=self.initializer)

    def call(self, x, training):
        w_bar = self.spectral_norm(self.kernel, training=training)
        x = tf.nn.conv2d(x, w_bar, strides=[1, 1, 1, 1], padding='SAME')
        return x


class SNLinear(layers.Layer):
    def __init__(self,
                 units,
                 use_bias=True,
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='snlinear'):
        super(SNLinear, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.initializer = initializer

    def build(self, input_shape):
        self.spectral_norm = SpectralNorm(num_iters=self.sn_iters)
        kernel_shape = [int(input_shape[1]), self.units]
        self.kernel = self.add_weight('kernel', shape=kernel_shape,
                                      initializer=self.initializer)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.units],
                                        initializer=tf.zeros_initializer())

    def call(self, x, training):
        w_bar = self.spectral_norm(self.kernel, training=training)
        x = tf.matmul(x, w_bar)
        if self.use_bias:
            x += self.bias
        return x


class Embedding(layers.Layer):
    def __init__(self,
                 num_classes,
                 embedding_size,
                 initializer=tf.initializers.orthogonal(),
                 name='embedding'):
        super(Embedding, self).__init__(name=name)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.initializer = initializer

    def build(self, input_shape):
        embed_shape = [self.num_classes, self.embedding_size]
        self.embed_map = self.add_weight('embed_map', shape=embed_shape,
                                         dtype=tf.float32,
                                         initializer=self.initializer)

    def call(self, x):
        x = tf.nn.embedding_lookup(self.embed_map, x)
        return x


class SNEmbedding(layers.Layer):
    def __init__(self,
                 num_classes,
                 embedding_size,
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='snembedding'):
        super(SNEmbedding, self).__init__(name=name)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.sn_iters = sn_iters
        self.initializer = initializer

    def build(self, input_shape):
        self.spectral_norm = SpectralNorm(num_iters=self.sn_iters)
        embed_shape = [self.num_classes, self.embedding_size]
        self.embed_map = self.add_weight('embed_map', shape=embed_shape,
                                         initializer=self.initializer)

    def call(self, x, training):
        embed_map_bar_T = self.spectral_norm(tf.transpose(self.embed_map),
                                             training=training)
        embed_map_bar = tf.transpose(embed_map_bar_T)
        x = tf.nn.embedding_lookup(embed_map_bar, x)
        return x


class SelfAttention(layers.Layer):
    def __init__(self,
                 initializer=tf.initializers.orthogonal(),
                 name='self_attention'):
        super(SelfAttention, self).__init__(name=name)
        self.initializer = initializer

    def build(self, input_shape):
        num_channels = int(input_shape[-1])
        self.conv_theta = SNConv1x1(num_channels//8,
                                    initializer=self.initializer,
                                    name='sn_conv_theta')
        self.conv_phi = SNConv1x1(num_channels//8,
                                  initializer=self.initializer,
                                  name='sn_conv_phi')
        self.conv_g = SNConv1x1(num_channels//2,
                                initializer=self.initializer,
                                name='sn_conv_g')
        self.conv_attn = SNConv1x1(num_channels,
                                   initializer=self.initializer,
                                   name='sn_conv_attn')
        self.sigma = self.add_weight('sigma', shape=[],
                                     initializer=tf.zeros_initializer())

    def call(self, x, training):
        batch_size, h, w, num_channels = map(int, x.shape.as_list())
        location_num = h*w
        downsampled_num = location_num//4

        theta = self.conv_theta(x, training=training)
        theta = tf.reshape(theta, [batch_size, location_num, num_channels//8])

        phi = self.conv_phi(x, training=training)
        phi = tf.nn.max_pool(phi, ksize=[2, 2], strides=2, padding='VALID')
        phi = tf.reshape(phi, [batch_size, downsampled_num, num_channels//8])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(x, training=training)
        g = tf.nn.max_pool(g, ksize=[2, 2], strides=2, padding='VALID')
        g = tf.reshape(g, [batch_size, downsampled_num, num_channels//2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels//2])
        attn_g = self.conv_attn(attn_g, training=training)

        return x + self.sigma * attn_g


class ConditionalBatchNorm(layers.Layer):
    def __init__(self,
                 input_dim,
                 axis=-1,
                 momentum=0.999,
                 epsilon=1E-5,
                 initializer=tf.initializers.orthogonal(),
                 name='conditional_batch_norm'):
        super(ConditionalBatchNorm, self).__init__(name=name)
        self.input_dim = input_dim
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.bn = tf.keras.layers.BatchNormalization(axis, momentum, epsilon,
                                                     center=False, scale=False,
                                                     name='bn')
        self.linear_beta = SNLinear(input_dim, use_bias=False,
                                    initializer=initializer,
                                    name='sn_linear_beta')
        self.linear_gamma = SNLinear(input_dim, use_bias=False,
                                     initializer=initializer,
                                     name='sn_linear_gamma')

    def call(self, x, condition, training):
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

    sn = SpectralNorm()
    conv2d = SNConv2d(filters)
    conv1x1 = SNConv1x1(filters)
    linear = SNLinear(units)
    embed0 = Embedding(num_classes, embedding_size)
    embed = SNEmbedding(num_classes, embedding_size)
    self_attn = SelfAttention()
    cbn = ConditionalBatchNorm(input_dim)

    # Create sample inputs
    weights = tf.random.normal((*kernel_size, input_dim, output_dim), dtype=tf.float32)
    images = tf.random.normal((batch_size, 64, 64, 3), dtype=tf.float32)
    features = tf.random.normal((batch_size, 100), dtype=tf.float32)
    labels = tf.random.uniform((batch_size, ), minval=0, maxval=10, dtype=tf.dtypes.int32)

    training = True
    # Singular value estimation of spectral normalization
    import numpy as np
    norm = np.linalg.norm
    import scipy as sp
    w_np = weights.numpy().reshape((-1, output_dim))
    u, s, v_t = sp.linalg.svd(w_np.T)
    u0 = u[:,0]
    for _ in range(100):
        _ = sn(weights, training=training)
    u_pseudo = sn.u.numpy()
    cossim = np.abs(np.sum(u0*u_pseudo)) / norm(u0) / norm(u_pseudo)
    if cossim > 0.99:
        print(f'Singular vector similarity: {cossim}')
    else:
        raise ValueError(f'Singular vector estimation failed, cosine similarity: {cossim}')
    # Forward into layers
    _ = conv2d(images, training=training)
    _ = conv1x1(images, training=training)
    _ = linear(features, training=training)
    _ = embed0(labels, training=training)
    _ = embed(labels, training=training)
    _ = self_attn(images, training=training)
    _ = cbn(images, features, training=training)
    
    training = False
    for _ in range(100):
        _ = sn(weights, training=training)
    u_fix = sn.u.numpy()
    cossim = np.abs(np.sum(u_pseudo*u_fix)) / norm(u_pseudo) / norm(u_fix)
    if cossim > 0.99:
        print(f'Singular vector preservation : {cossim}')
    else:
        raise ValueError(f'Singular vector preservation failed, cosine similarity: {cossim}')
    _ = conv2d(images, training=training)
    _ = conv1x1(images, training=training)
    _ = linear(features, training=training)
    _ = embed0(labels, training=training)
    _ = embed(labels, training=training)
    _ = self_attn(images, training=training)
    _ = cbn(images, features, training=training)

    print('Completed.')
