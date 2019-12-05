import tensorflow as tf
from tensorflow.keras import layers

from . import ops


def usample(x):
    _, h, w, ch = x.shape.as_list()
    x = tf.image.resize(x, [2*h, 2*w], method='nearest')
    return x


class GBlock(layers.Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 name='block'):
        super(GBlock, self).__init__(name=name)
        self.input_dim = input_dim
        self.filters = filters

        self.cbn1 = ops.ConditionalBatchNorm(input_dim, name='cbn_1')
        self.cbn2 = ops.ConditionalBatchNorm(filters, name='cbn_2')
        self.conv1 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_1')
        self.conv2 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_2')
        self.conv3 = ops.SNConv2d(filters, (1, 1), (1, 1), name='snconv_3')

    def call(self, inputs, training=None):
        x, condition = inputs
        fx = tf.nn.relu(self.cbn1([x, condition], training=training))
        fx = usample(fx)
        fx = self.conv1(fx, training=training)
        fx = tf.nn.relu(self.cbn2([fx, condition], training=training))
        fx = self.conv2(fx, training=training)

        x = usample(x)
        x = self.conv3(x, training=training)

        return x + fx


class Generator(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 gf_dim=64,
                 embedding_size=128,
                 name='generator'):
        super(Generator, self).__init__(name=name)
        self.num_classes = num_classes
        self.gf_dim = gf_dim
        self.embedding_size = embedding_size

        self.embed = layers.Embedding(num_classes, embedding_size,
                                      name='embed')
        self.linear = ops.SNLinear(gf_dim*8*4*4, name='linear')
        self.block1 = GBlock(gf_dim*8, gf_dim*8, name='block_1')
        self.block2 = GBlock(gf_dim*8, gf_dim*4, name='block_2')
        self.block3 = GBlock(gf_dim*4, gf_dim*2, name='block_3')
        self.attn = ops.SelfAttention(name='attn')
        self.block4 = GBlock(gf_dim*2, gf_dim, name='block_4')

        self.bn_out = layers.BatchNormalization(momentum=0.9999,
                                                epsilon=1e-5,
                                                name='bn_out')
        self.conv_out = ops.SNConv2d(3, (3, 3), (1, 1), name='conv_out')

    def call(self, inputs, training=None):
        z, target_class = inputs
        z_split = tf.split(z, num_or_size_splits=5, axis=-1)
        embed = self.embed(target_class)
        conds = [tf.concat([z_i, embed], axis=-1) for z_i in z_split[1:]]

        x = self.linear(z_split[0], training=training)
        x = tf.reshape(x, [-1, 4, 4, self.gf_dim*8])
        x = self.block1([x, conds[0]], training=training)
        x = self.block2([x, conds[1]], training=training)
        x = self.block3([x, conds[2]], training=training)
        x = self.attn(x, training=training)
        x = self.block4([x, conds[3]], training=training)
        
        x = tf.nn.relu(self.bn_out(x, training=training))
        x = self.conv_out(x, training=training)
        x = tf.nn.tanh(x)
        return x

    
class DCGenerator(tf.keras.Model):
    """ Basic DCGAN (could have a few differences from the original one) """
    def __init__(self,
                 gf_dim=64,
                 name='generator'):
        super().__init__(name=name)
        self.gf_dim = gf_dim

        self.seq = tf.keras.Sequential([
            layers.Dense(gf_dim*8*4*4, use_bias=False, name='fc'),
            layers.Reshape((4, 4, gf_dim*8)),

            layers.BatchNormalization(name='bn'),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(gf_dim*4, (5, 5), padding='same'),
            
            layers.BatchNormalization(name='bn'),
            layers.ReLU(),
            layers.UpSampling2D(),
            layers.Conv2D(gf_dim*2, (5, 5), padding='same'),

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
    def __init__(self,
                 gf_dim=64,
                 name='generator'):
        super(ResNetGenerator, self).__init__(name=name)
        self.gf_dim = gf_dim

        self.fc = layers.Dense(gf_dim*8*4*4, name='fc')
        self.block1 = self._block(gf_dim*8, name='block_1')
        self.sc1 = layers.Conv2D(gf_dim*8, (1, 1), name='sc_1')
        self.block2 = self._block(gf_dim*4, name='block_2')
        self.sc2 = layers.Conv2D(gf_dim*4, (1, 1), name='sc_2')
        self.block3 = self._block(gf_dim*2, name='block_3')
        self.sc3 = layers.Conv2D(gf_dim*2, (1, 1), name='sc_3')
        self.block4 = self._block(gf_dim, name='block_4')
        self.sc4 = layers.Conv2D(gf_dim, (1, 1), name='sc_4')

        self.bn = layers.BatchNormalization(momentum=0.9999, epsilon=1e-5,
                                            name='bn')
        self.conv = layers.Conv2D(3, (3, 3), (1, 1), 'same', name='conv')

    def call(self, x, training=None):
        x = self.fc(x)
        x = tf.reshape(x, [-1, 4, 4, self.gf_dim*8])
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
        ], name=name)
        return seq


def dsample(x):
    x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    return x


class DBlock(layers.Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 downsample=True,
                 name='block'):
        super(DBlock, self).__init__(name=name)
        self.input_dim = input_dim
        self.filters = filters
        self.downsample = downsample

        self.conv1 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_1')
        self.conv2 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_2')
        if downsample:
            self.conv3 = ops.SNConv2d(filters, (1, 1), (1, 1), name='snconv_3')

    def call(self, x, training=None):
        fx = tf.nn.relu(x)
        fx = self.conv1(fx, training=training)
        fx = tf.nn.relu(fx)
        fx = self.conv2(fx, training=training)
        
        if self.downsample:
            fx = dsample(fx)
            x = self.conv3(x, training=training)
            x = dsample(x)
        return x + fx


class Discriminator(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 df_dim=64,
                 name='discriminator'):
        super(Discriminator, self).__init__(name=name)
        self.num_classes = num_classes
        self.df_dim = df_dim

        self.block1 = DBlock(3, df_dim, name='block_1')
        self.attn = ops.SelfAttention(name='attn')
        self.block2 = DBlock(df_dim, df_dim*2, name='block_2')
        self.block3 = DBlock(df_dim*2, df_dim*4, name='block_3')
        self.block4 = DBlock(df_dim*4, df_dim*8, name='block_4')
        self.block5 = DBlock(df_dim*8, df_dim*8, downsample=False,
                             name='block_5')
        
        self.linear = ops.SNLinear(1, name='linear_out')
        self.embed = ops.SNEmbedding(num_classes, df_dim*8,
                                     name='embed')

    def call(self, inputs, training=True):
        x, labels = inputs
        x = self.block1(x, training=training)
        x = self.attn(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        
        x = tf.nn.relu(x)
        x = tf.reduce_sum(x, axis=[1, 2])
        out = self.linear(x, training=training)
        embed = self.embed(labels)
        out += tf.reduce_sum(x*embed, axis=-1, keepdims=True)
        return out

    
class DCDiscriminator(tf.keras.Model):
    """ https://www.tensorflow.org/tutorials/generative/dcgan """
    def __init__(self,
                 df_dim=64,
                 name='discriminator'):
        super().__init__(name=name)
        self.df_dim = df_dim

        self.seq = tf.keras.Sequential([
            layers.Conv2D(df_dim, (5, 5), (2, 2), 'same'),
            layers.LeakyReLU(),
            layers.Conv2D(df_dim*2, (5, 5), (2, 2), 'same'),
            layers.LeakyReLU(),
            layers.Conv2D(df_dim*4, (5, 5), (2, 2), 'same'),
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
    def __init__(self,
                 df_dim=64,
                 name='discriminator'):
        super(ResNetDiscriminator, self).__init__(name=name)
        self.df_dim = df_dim

        self.block1 = self._block(df_dim, name='block_1')
        self.sc1 = layers.Conv2D(df_dim, (1, 1), name='sc_1')
        self.block2 = self._block(df_dim*2, name='block_2')
        self.sc2 = layers.Conv2D(df_dim*2, (1, 1), name='sc_2')
        self.block3 = self._block(df_dim*4, name='block_3')
        self.sc3 = layers.Conv2D(df_dim*4, (1, 1), name='sc_3')
        self.block4 = self._block(df_dim*8, name='block_4')
        self.sc4 = layers.Conv2D(df_dim*8, (1, 1), name='sc_4')
        self.block5 = self._block(df_dim*8, name='block_5')
        self.sc5 = layers.Conv2D(df_dim*8, (1, 1), name='sc_5')

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
        ], name=name)
        return seq


if __name__ == '__main__':
    batch_size = 4
    z_dim = 120
    image_size = (64, 64)
    num_classes = 10
    gf_dim = 64
    df_dim = 64
    embedding_size = 128

    # sample definition
    z = tf.random.normal((batch_size, z_dim), dtype=tf.float32)
    images = tf.random.normal((batch_size, *image_size, 3), dtype=tf.float32)
    labels = tf.random.uniform((batch_size, ), minval=0, maxval=num_classes,
                               dtype=tf.int32)

    # create model
    generator = Generator(num_classes, gf_dim, embedding_size,
                          name='generator')
    discriminator = Discriminator(num_classes, df_dim, name='discriminator')

    training = True
    print('Validating training loop ...')
    for _ in range(10):
        images_fake = generator([z, labels], training=training)
        logits_real = discriminator([images, labels], training=training)
        
    training = False
    print('Validating test loop ...')
    for _ in range(10):
        image_fake = generator([z, labels], training=training)
        logits_fake = discriminator([images_fake, labels], training=training)

    print('-------------- Completed -----------------')
