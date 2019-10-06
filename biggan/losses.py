import tensorflow as tf


def get_d_real_loss(logits_real):
    loss = tf.nn.relu(1.0 - logits_real)
    return tf.reduce_mean(loss)

def get_d_fake_loss(logits_fake):
    loss = tf.nn.relu(1.0 + logits_fake)
    return tf.reduce_mean(loss)

def get_g_loss(logits_fake):
    return -tf.reduce_mean(logits_fake)
