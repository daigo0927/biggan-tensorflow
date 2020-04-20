import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, losses, metrics
import ops


def validate_stectral_norm():
    sample_inputs = tf.zeros((1, 32, 32, 3))
    filters = 16
    snconv = ops.SNConv2D(filters, 3, 2, padding='same')

    for _ in range(100):
        _ = snconv(sample_inputs, training=True)

    weight = snconv.layer.kernel
    # Spectral norm (1st sigular vector) computed by scipy
    w_np = weight.numpy().reshape((-1, filters))
    u_np, s_np, vt_np = sp.linalg.svd(w_np.T)
    u0_np = u_np[:, 0]
    # Spectral norm computed by the layer
    u_pseudo = snconv.u.numpy()[:, 0]
    cossim = np.abs(np.sum(
        u0_np * u_pseudo)) / np.linalg.norm(u0_np) / np.linalg.norm(u_pseudo)
    if cossim > 0.95:
        print(f'Singular vector similarity: {cossim}')
    else:
        raise ValueError(
            f'Singular vector estimation failed, cosine similarity: {cossim}')


def preprocess(image, label):
    image = tf.cast(image, dtype=tf.float32) / 255
    label = tf.cast(label, dtype=tf.int32)
    return image, label


def train():
    epochs = 1
    batch_size = 32

    (ds_train, ds_test), info = tfds.load('cifar10',
                                          split=['train', 'test'],
                                          as_supervised=True,
                                          with_info=True)
    print(info)

    n_train_samples = 50000
    n_test_samples = 10000
    n_classes = 10
    ds_train = ds_train.shuffle(n_train_samples)\
        .batch(batch_size)\
        .map(preprocess)\
        .repeat(epochs)
    ds_test = ds_test.map(preprocess).batch(batch_size)

    model = tf.keras.Sequential([
        ops.SNConv2D(32, 3, 2, padding='same', activation='relu'),
        ops.SNConv2D(32, 3, 1, padding='same', activation='relu'),
        ops.SNSelfAttention(),
        ops.SNConv2D(64, 3, 2, padding='same', activation='relu'),
        ops.SNConv2D(64, 3, 1, padding='same', activation='relu'),
        layers.Flatten(),
        ops.SNLinear(n_classes)
    ])
    model.build(input_shape=[None, 32, 32, 3])
    model.summary()

    model.compile(optimizer='sgd',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')])
    model.fit(ds_train,
              epochs=epochs,
              validation_data=ds_test,
              steps_per_epoch=n_train_samples // batch_size,
              validation_steps=n_test_samples // batch_size)


if __name__ == '__main__':
    validate_stectral_norm()
    train()
