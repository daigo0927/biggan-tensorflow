import tensorflow as tf
from tensorflow.keras import layers, initializers

from . import ops


class GBlock(layers.Layer):
    def __init__(self,
                 filters,
                 upsample=True,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.upsample = upsample
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.u_initializer = u_initializer

        self._init_params = {
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'u_initializer': u_initializer
        }

    def build(self, input_shape):
        self.bn_1 = ops.ConditionalBatchNorm(use_bias=False,
                                             name='bn_1',
                                             **self._init_params)
        self.bn_2 = ops.ConditionalBatchNorm(use_bias=False,
                                             name='bn_2',
                                             **self._init_params)
        self.activation = layers.ReLU()
        if self.upsample:
            self.upsampling = layers.UpSampling2D()

        self.conv_1 = ops.SNConv2D(self.filters, (3, 3), (1, 1),
                                   name='snconv_1',
                                   **self._init_params)
        self.conv_2 = ops.SNConv2D(self.filters, (3, 3), (1, 1),
                                   name='snconv_2',
                                   **self._init_params)
        self.conv_sc = ops.SNConv2D(self.filters, (1, 1), (1, 1),
                                    name='snconv_sc',
                                    **self._init_params)

    def call(self, inputs, training=None):
        x, condition = inputs
        fx = self.activation(self.bn_1([x, condition], training))
        if self.upsample:
            fx = self.upsampling(fx)
            x = self.upsampling(x)

        fx = self.conv_1(fx, training)
        fx = self.activation(self.bn_2([fx, condition], training))
        fx = self.conv_2(fx, training)

        x = self.conv_sc(x, training)
        return x + fx


class Generator(tf.keras.Model):
    def __init__(self, num_classes, base_dim=64, embedding_size=100, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.base_dim = base_dim
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.embed = layers.Embedding(self.num_classes,
                                      self.embedding_size,
                                      name='embed')
        self.linear = ops.SNLinear(self.base_dim * 8 * 4 * 4, name='linear')
        self.block_1 = GBlock(self.base_dim * 8, name='block_1')
        self.block_2 = GBlock(self.base_dim * 4, name='block_2')
        self.block_3 = GBlock(self.base_dim * 2, name='block_3')
        self.attn = ops.SNSelfAttention(use_bias=False, name='attn')
        self.block_4 = GBlock(self.base_dim, name='block_4')

        self.out = tf.keras.Sequential([
            layers.BatchNormalization(
                momentum=0.9999, epsilon=1e-5, name='bn_out'),
            layers.ReLU(),
            ops.SNConv2D(3, (3, 3), (1, 1), name='conv_out'),
            layers.Activation('tanh')
        ],
                                       name='out')

    def call(self, inputs, training=None):
        z, y = inputs
        zs = tf.split(z, num_or_size_splits=5, axis=-1)
        embed = self.embed(y)
        conditions = [tf.concat([embed, z_i], axis=-1) for z_i in zs[1:]]

        x = self.linear(zs[0], training)
        x = tf.reshape(x, [-1, 4, 4, self.base_dim * 8])
        x = self.block_1([x, conditions[0]], training)
        x = self.block_2([x, conditions[1]], training)
        x = self.block_3([x, conditions[2]], training)
        x = self.attn(x, training)
        x = self.block_4([x, conditions[3]], training)
        return self.out(x, training)


class DCGenerator(tf.keras.Model):
    """ Basic DCGAN (could have a few differences from the original one) """
    def __init__(self, gf_dim=64, name='generator'):
        super().__init__(name=name)
        self.gf_dim = gf_dim

        self.seq = tf.keras.Sequential([
            layers.Dense(gf_dim * 8 * 4 * 4, use_bias=False, name='fc'),
            layers.Reshape((4, 4, gf_dim * 8)),
            layers.BatchNormalization(name='bn'),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(gf_dim * 4, (5, 5), padding='same'),
            layers.BatchNormalization(name='bn'),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(gf_dim * 2, (5, 5), padding='same'),
            layers.BatchNormalization(name='bn'),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(gf_dim, (5, 5), padding='same'),
            layers.BatchNormalization(name='bn'),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(3, (5, 5), padding='same'),
            layers.Activation('tanh')
        ])

    def call(self, inputs, training=None):
        outputs = self.seq(inputs, training=training)
        return outputs


class ResNetGenerator(tf.keras.Model):
    def __init__(self, gf_dim=64, name='generator'):
        super(ResNetGenerator, self).__init__(name=name)
        self.gf_dim = gf_dim

        self.fc = layers.Dense(gf_dim * 8 * 4 * 4, name='fc')
        self.block1 = self._block(gf_dim * 8, name='block_1')
        self.sc1 = layers.Conv2D(gf_dim * 8, (1, 1), name='sc_1')
        self.block2 = self._block(gf_dim * 4, name='block_2')
        self.sc2 = layers.Conv2D(gf_dim * 4, (1, 1), name='sc_2')
        self.block3 = self._block(gf_dim * 2, name='block_3')
        self.sc3 = layers.Conv2D(gf_dim * 2, (1, 1), name='sc_3')
        self.block4 = self._block(gf_dim, name='block_4')
        self.sc4 = layers.Conv2D(gf_dim, (1, 1), name='sc_4')

        self.bn = layers.BatchNormalization(momentum=0.9999,
                                            epsilon=1e-5,
                                            name='bn')
        self.conv = layers.Conv2D(3, (3, 3), (1, 1), 'same', name='conv')

    def call(self, x, training=None):
        x = self.fc(x)
        x = tf.reshape(x, [-1, 4, 4, self.gf_dim * 8])
        fx = self.block1(x, training=training)
        x = self.sc1(usample(x)) + fx
        fx = self.block2(x, training=training)
        x = self.sc2(usample(x)) + fx
        fx = self.block3(x, training=training)
        x = self.sc3(usample(x)) + fx
        fx = self.block4(x, training=training)
        x = self.sc4(usample(x)) + fx

        x = tf.nn.relu(self.bn(x, training=training))
        x = tf.nn.tanh(self.conv(x))
        return x

    def _block(self, filters, name):
        seq = tf.keras.Sequential([
            layers.BatchNormalization(name='bn_1'),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(filters, (3, 3), (1, 1), 'same', name='conv_1'),
            layers.BatchNormalization(name='bn_2'),
            layers.ReLU(),
            layers.Conv2D(filters, (3, 3), (1, 1), 'same', name='conv_2')
        ],
                                  name=name)
        return seq


class DBlock(layers.Layer):
    def __init__(self,
                 filters,
                 downsample=True,
                 preactivation=True,
                 kernel_initializer=initializers.Orthogonal(),
                 bias_initializer=initializers.Zeros(),
                 u_initializer=initializers.RandomNormal(),
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.downsample = downsample
        self.preactivation = preactivation

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.u_initializer = u_initializer

        self._init_params = {
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'u_initializer': u_initializer
        }

    def build(self, input_shape):
        self.activation = layers.ReLU()
        if self.downsample:
            self.downsampling = layers.AvgPool2D(pool_size=(2, 2))

        self.conv_1 = ops.SNConv2D(self.filters, (3, 3), (1, 1),
                                   name='snconv_1',
                                   **self._init_params)
        self.conv_2 = ops.SNConv2D(self.filters, (3, 3), (1, 1),
                                   name='snconv_2',
                                   **self._init_params)
        self.conv_sc = ops.SNConv2D(self.filters, (1, 1), (1, 1),
                                    name='snconv_sc',
                                    **self._init_params)

    def call(self, inputs, training=None):
        if self.preactivation:
            x = self.activation(inputs)
        else:
            x = inputs

        fx = self.conv_1(x, training)
        fx = self.conv_2(self.activation(fx), training)

        if self.downsample:
            fx = self.downsampling(fx)
            x = self.downsampling(x)

        x = self.conv_sc(x, training)
        return x + fx


class Discriminator(tf.keras.Model):
    def __init__(self, num_classes, base_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.base_dim = base_dim

    def build(self, inputs_shape):
        self.block_1 = DBlock(self.base_dim,
                              preactivation=False,
                              name='block_1')
        self.attn = ops.SNSelfAttention(name='attn')
        self.block_2 = DBlock(self.base_dim * 2, name='block_2')
        self.block_3 = DBlock(self.base_dim * 4, name='block_3')
        self.block_4 = DBlock(self.base_dim * 8, name='block_4')
        self.block_5 = DBlock(self.base_dim * 16,
                              downsample=False,
                              name='block_5')

        self.activation = layers.ReLU()

        self.linear = ops.SNLinear(1, name='linear_out')
        self.embed = ops.SNEmbedding(num_classes,
                                     self.base_dim * 16,
                                     name='embed')

    def call(self, inputs, training=True):
        x, labels = inputs
        x = self.block_1(x, training)
        x = self.attn(x, training)
        x = self.block_2(x, training)
        x = self.block_3(x, training)
        x = self.block_4(x, training)
        x = self.block_5(x, training)

        x = self.activation(x)
        x = tf.reduce_sum(x, axis=[1, 2])
        out = self.linear(x, training)
        embed = self.embed(labels, training)
        out += tf.reduce_sum(x * embed, axis=-1, keepdims=True)
        return out


class DCDiscriminator(tf.keras.Model):
    """ https://www.tensorflow.org/tutorials/generative/dcgan """
    def __init__(self, df_dim=64, name='discriminator'):
        super().__init__(name=name)
        self.df_dim = df_dim

        self.seq = tf.keras.Sequential([
            layers.Conv2D(df_dim, (5, 5), (2, 2), 'same'),
            layers.LeakyReLU(),
            layers.Conv2D(df_dim * 2, (5, 5), (2, 2), 'same'),
            layers.LeakyReLU(),
            layers.Conv2D(df_dim * 4, (5, 5), (2, 2), 'same'),
            layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    def call(self, inputs, training=None):
        outputs = self.seq(inputs, training=training)
        return outputs


class ResNetDiscriminator(tf.keras.Model):
    def __init__(self, df_dim=64, name='discriminator'):
        super(ResNetDiscriminator, self).__init__(name=name)
        self.df_dim = df_dim

        self.block1 = self._block(df_dim, name='block_1')
        self.sc1 = layers.Conv2D(df_dim, (1, 1), name='sc_1')
        self.block2 = self._block(df_dim * 2, name='block_2')
        self.sc2 = layers.Conv2D(df_dim * 2, (1, 1), name='sc_2')
        self.block3 = self._block(df_dim * 4, name='block_3')
        self.sc3 = layers.Conv2D(df_dim * 4, (1, 1), name='sc_3')
        self.block4 = self._block(df_dim * 8, name='block_4')
        self.sc4 = layers.Conv2D(df_dim * 8, (1, 1), name='sc_4')
        self.block5 = self._block(df_dim * 8, name='block_5')
        self.sc5 = layers.Conv2D(df_dim * 8, (1, 1), name='sc_5')

        self.fc = layers.Dense(1, name='fc')

    def call(self, x, training=None):
        fx = self.block1(x, training=training)
        x = dsample(self.sc1(x)) + dsample(fx)
        fx = self.block2(x, training=training)
        x = dsample(self.sc2(x)) + dsample(fx)
        fx = self.block3(x, training=training)
        x = dsample(self.sc3(x)) + dsample(fx)
        fx = self.block4(x, training=training)
        x = dsample(self.sc4(x)) + dsample(fx)
        fx = self.block5(x, training=training)
        x = self.sc5(x) + dsample(fx)

        x = tf.nn.relu(x)
        x = tf.reduce_sum(x, axis=(1, 2))
        x = self.fc(x)
        return x

    def _block(self, filters, name):
        seq = tf.keras.Sequential([
            layers.ReLU(),
            layers.Conv2D(filters, (3, 3), (1, 1), 'same', name='conv_1'),
            layers.ReLU(),
            layers.Conv2D(filters, (3, 3), (1, 1), 'same', name='conv_2')
        ],
                                  name=name)
        return seq


class BigGAN64(tf.keras.Model):
    def __init__(self, latent_dim, num_classes, generator, discriminator,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def train_step(self, inputs_real):
        images_real, labels_real = inputs_real

        batch_size = tf.shape(images_real)[0]

        zs = tf.random.normal(shape=(batch_size, self.latent_dim))
        labels_fake = tf.random.uniform((batch_size, ),
                                        0,
                                        self.num_classes,
                                        dtype=tf.int32)
        images_fake = self.generator([zs, labels_fake], training=False)

        images = tf.concat([images_real, images_fake], axis=0)
        labels = tf.concat([labels_real, labels_fake], axis=0)
        with tf.GradientTape() as tape:
            logits = self.discriminator([images, labels], training=True)
            logits_real, logits_fake = tf.split(logits, 2, axis=0)
            d_loss = self.d_loss_fn(logits_real, logits_fake)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))

        zs = tf.random.normal(shape=(batch_size, self.latent_dim))
        labels_fake = tf.random.uniform((batch_size, ),
                                        0,
                                        self.num_classes,
                                        dtype=tf.int32)
        with tf.GradientTape() as tape:
            images_fake = self.generator([zs, labels_fake], training=True)
            logits_fake = self.discriminator([images_fake, labels_fake],
                                             training=False)
            g_loss = self.g_loss_fn(logits_fake)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        return {'d_loss': d_loss, 'g_loss': g_loss}


if __name__ == '__main__':
    batch_size = 1
    z_dim = 100
    image_size = (64, 64)
    num_classes = 10
    base_dim = 64
    embedding_size = 128

    # sample definition
    images = tf.random.normal((batch_size, *image_size, 3), dtype=tf.float32)
    labels = tf.random.uniform((batch_size, ),
                               minval=0,
                               maxval=num_classes,
                               dtype=tf.int32)

    # create model
    generator = Generator(num_classes,
                          base_dim,
                          embedding_size,
                          name='generator')
    discriminator = Discriminator(num_classes, base_dim, name='discriminator')
    biggan = BigGAN64(z_dim,
                      num_classes,
                      generator,
                      discriminator,
                      name='biggan')

    from losses import d_hinge_loss, g_hinge_loss
    biggan.compile(g_optimizer=tf.keras.optimizers.Adam(),
                   d_optimizer=tf.keras.optimizers.Adam(),
                   g_loss_fn=g_hinge_loss,
                   d_loss_fn=d_hinge_loss)

    biggan.fit(images, labels, batch_size=1, epochs=1)

    # @tf.function
    # def train_step(images_real, labels_real):
    #     losses = biggan.train_step([images_real, labels_real])
    #     return losses

    # from datetime import datetime
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = 'logs_test/func/%s' % stamp
    # writer = tf.summary.create_file_writer(logdir)

    # tf.summary.trace_on(graph=True, profiler=True)
    # outputs = train_step(images, labels)
    # with writer.as_default():
    #     tf.summary.trace_export(name='biggan_trace',
    #                             step=0,
    #                             profiler_outdir=logdir)
    print('-------------- Completed -----------------')
