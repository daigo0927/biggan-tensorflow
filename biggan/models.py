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
                 name):
        super(GBlock, self).__init__(name=name)
        self.input_dim = input_dim
        self.filters = filters

        self.cbn0 = ops.ConditionalBatchNorm(input_dim, name='cbn_0')
        self.cbn1 = ops.ConditionalBatchNorm(filters, name='cbn_1')
        self.conv0 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_0')
        self.conv1 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_1')
        self.conv2 = ops.SNConv2d(filters, (1, 1), (1, 1), name='snconv_2')

    def call(self, x, condition, training):
        x_0 = x
        x = tf.nn.relu(self.cbn0(x, condition, training=training))
        x = usample(x)
        x = self.conv0(x, training=training)
        x = tf.nn.relu(self.cbn1(x, condition, training=training))
        x = self.conv1(x, training=training)

        x_0 = usample(x_0)
        x_0 = self.conv2(x_0, training=training)

        return x_0 + x


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

        self.embed = ops.Embedding(num_classes, embedding_size, name='embed')
        self.linear = ops.SNLinear(gf_dim*16*4*4, name='linear')
        self.block0 = GBlock(gf_dim*16, gf_dim*16, name='block_0')
        self.block1 = GBlock(gf_dim*16, gf_dim*8, name='block_1')
        self.block2 = GBlock(gf_dim*8, gf_dim*4, name='block_2')
        self.attn = ops.SelfAttention(name='attn')
        self.block3 = GBlock(gf_dim*4, gf_dim*2, name='block_3')
        self.block4 = GBlock(gf_dim*2, gf_dim, name='block_4')

        self.bn_out = tf.keras.layers.BatchNormalization(momentum=0.9999,
                                                         epsilon=1e-5,
                                                         name='bn_out')
        self.conv_out = ops.SNConv2d(3, (3, 3), (1, 1), name='conv_out')

    def call(self, z, target_class, training=True):
        z_split = tf.split(z, num_or_size_splits=6, axis=-1)
        z_0 = z_split[0]
        embed = self.embed(target_class)
        conds = [tf.concat([z_i, embed], axis=-1) for z_i in z_split[1:]]

        x = self.linear(z_0, training=training)
        x = tf.reshape(x, [-1, 4, 4, self.gf_dim*16])
        x = self.block0(x, conds[0], training=training)
        x = self.block1(x, conds[1], training=training)
        x = self.block2(x, conds[2], training=training)
        x = self.attn(x, training=training)
        x = self.block3(x, conds[3], training=training)
        x = self.block4(x, conds[4], training=training)
        
        x = tf.nn.relu(self.bn_out(x, training=training))
        x = self.conv_out(x, training=training)
        x = tf.nn.tanh(x)
        return x


def dsample(x):
    x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    return x


class DBlock(layers.Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 name,
                 downsample=True):
        super(DBlock, self).__init__(name=name)
        self.input_dim = input_dim
        self.filters = filters
        self.downsample = downsample

        self.conv0 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_0')
        self.conv1 = ops.SNConv2d(filters, (3, 3), (1, 1), name='snconv_1')
        if downsample:
            self.conv2 = ops.SNConv2d(filters, (1, 1), (1, 1), name='snconv_2')

    def call(self, x, training):
        x_0 = x
        x = tf.nn.relu(x)
        x = self.conv0(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x, training=training)
        
        if self.downsample:
            x = dsample(x)
            x_0 = self.conv2(x_0, training=training)
            x_0 = dsample(x_0)
        return x_0 + x


class Discriminator(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 df_dim=64,
                 name='discriminator'):
        super(Discriminator, self).__init__(name=name)
        self.num_classes = num_classes
        self.df_dim = df_dim

        self.block0 = DBlock(3, df_dim, name='block_0')
        self.block1 = DBlock(df_dim, df_dim*2, name='block_1')
        self.attn = ops.SelfAttention(name='attn')
        self.block2 = DBlock(df_dim*2, df_dim*4, name='block_2')
        self.block3 = DBlock(df_dim*4, df_dim*8, name='block_3')
        self.block4 = DBlock(df_dim*8, df_dim*16, name='block_4')
        self.block5 = DBlock(df_dim*16, df_dim*16, name='block_5',
                             downsample=False)
        
        self.linear = ops.SNLinear(1, name='linear_out')
        self.embed = ops.Embedding(num_classes, df_dim*16, name='embed')

    def call(self, x, labels, training=True):
        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.attn(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        
        x = tf.nn.relu(x)
        x = tf.reduce_sum(x, [1, 2])
        out = self.linear(x, training=training)
        embed = self.embed(labels)
        out += tf.reduce_sum(x*embed, axis=1, keepdims=True)
        return out


if __name__ == '__main__':
    batch_size = 4
    z_dim = 120
    image_size = (128, 128)
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
        images_fake = generator(z, labels, training=training)
        logits_real = discriminator(images, labels, training=training)
        
    training = False
    print('Validating test loop ...')
    for _ in range(10):
        image_fake = generator(z, labels, training=training)
        logits_fake = discriminator(images_fake, labels, training=training)

    print('-------------- Completed -----------------')
