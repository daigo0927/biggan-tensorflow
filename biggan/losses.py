import tensorflow as tf


def discriminator_hinge_loss(logits_real, logits_fake):
    loss_real = tf.nn.relu(1.0 - logits_real)
    loss_fake = tf.nn.relu(1.0 + logits_fake)
    return tf.reduce_mean(loss_real + loss_fake)


def generator_hinge_loss(logits_fake):
    return -tf.reduce_mean(logits_fake)


def bce(y_true, y_pred, from_logits=True, label_smoothing=0):
    loss = tf.losses.binary_crossentropy(y_true, y_pred,
                                         from_logits, label_smoothing)
    return loss


def discriminator_bce_loss(logits_real, logits_fake):
    loss_real = bce(tf.ones_like(logits_real), logits_real)
    loss_fake = bce(tf.zeros_like(logits_fake), logits_fake)
    return tf.reduce_mean(loss_real + loss_fake)


def generator_bce_loss(logits_fake):
    loss = bce(tf.ones_like(logits_fake), logits_fake)
    return tf.reduce_mean(loss)
