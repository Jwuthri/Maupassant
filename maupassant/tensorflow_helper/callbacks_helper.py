import tensorflow as tf


def frobenius_norm(m):
    """This function is to calculate the frobenius norm of the matrix of all layer's weight."""
    total_reduce_sum = 0
    for i in range(len(m)):
        total_reduce_sum = total_reduce_sum + tf.math.reduce_sum(m[i] ** 2)
    norm = total_reduce_sum ** 0.5
    return norm


def frobenius_normization_weights(model):
    """https://www.tensorflow.org/addons/tutorials/optimizers_conditionalgradient"""
    normalized = []

    return tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: normalized.append(frobenius_norm(model.trainable_weights).numpy()))
