from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
import tensorflow as tf


def frobenius_callback(model):
    """https://www.tensorflow.org/addons/tutorials/optimizers_conditionalgradient"""

    def normalize(m):
        """This function is to calculate the frobenius norm of the matrix of all layer's weight."""
        total_reduce_sum = 0
        for i in range(len(m)):
            total_reduce_sum = total_reduce_sum + tf.math.reduce_sum(m[i] ** 2)
        norm = total_reduce_sum ** 0.5

        return norm

    normalized = []
    weights = model.trainable_weights

    return LambdaCallback(on_epoch_end=lambda batch, logs: normalized.append(normalize(weights).numpy()))


def checkpoint_callback(ckpt_path):
    """Create basic callbacks for tensorflow."""
    return ModelCheckpoint(filepath=ckpt_path, verbose=1, period=1, save_weights_only=True)


def tensorboard_callback(tsboard_path=None):
    """Create basic callbacks for tensorflow."""
    return TensorBoard(log_dir=tsboard_path, histogram_freq=1)

