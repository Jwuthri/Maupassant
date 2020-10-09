import tensorflow as tf

from maupassant.feature_extraction.pretrained_embedding import PretrainedEmbedding


def get_input_layer(use_pretrained, embedding_size, input_size, vocab_size):
    """Create the input layer, with embedding."""
    if use_pretrained:
        input_layer = tf.keras.Input((), dtype=tf.string)
        layer = PretrainedEmbedding().model(input_layer)
        layer = tf.keras.layers.Reshape(target_shape=(1, 512))(layer)
    else:
        input_layer = tf.keras.Input((input_size))
        layer = tf.keras.layers.Embedding(vocab_size, embedding_size)(input_layer)

    return input_layer, layer

def get_output_layer(label_type, units=1, use_time_distrib=False):
    """Get the output layer, with the activation function based on the label_type."""
    if label_type == "binary-class":
        output = tf.keras.layers.Dense(units=1, activation="sigmoid")
    elif label_type == "multi-label":
        output = tf.keras.layers.Dense(units=units, activation="sigmoid")
    elif label_type == "multi-class":
        output = tf.keras.layers.Dense(units=units, activation="softmax")
    else:
        raise (Exception("Please provide a 'label_type' in ['binary-class', 'multi-label', 'multi-class']"))
    if use_time_distrib:
        output = tf.keras.layers.TimeDistributed(output)

    return output

def text_to_layer(block, unit, return_sequences=False):
    """Build tensorflow layer, easily."""
    layer = None
    if block == "CNN":
        layer = tf.keras.layers.Conv1D(unit, kernel_size=1, strides=1, padding='same', activation='relu')
    elif block == "LCNN":
        layer = tf.keras.layers.LocallyConnected1D(unit, kernel_size=1, strides=1, padding='valid', activation='relu')
    elif block == "BiLSTM":
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(unit, activation="relu", return_sequences=return_sequences))
    elif block == "BiGRU":
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(unit, activation="relu", return_sequences=return_sequences))
    elif block == "BiRNN":
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(unit, activation="relu", return_sequences=return_sequences))
    elif block == "CudaLSTM":
        layer = tf.compat.v1.keras.layers.CuDNNLSTM(unit, return_sequences=return_sequences)
    elif block == "LSTM":
        layer = tf.keras.layers.LSTM(unit, activation='relu', return_sequences=return_sequences)
    elif block == "GRU":
        layer = tf.keras.layers.GRU(unit, activation='relu', return_sequences=return_sequences)
    elif block == "RNN":
        layer = tf.keras.layers.SimpleRNN(unit, activation='relu', return_sequences=return_sequences)
    elif block == "DENSE":
        layer = tf.keras.layers.Dense(unit, activation="relu")
    elif block == "TIME_DISTRIB_DENSE":
        layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(unit, activation="relu"))
    elif block == "FLATTEN":
        layer = tf.keras.layers.Flatten()
    elif block == "RESHAPE":
        layer = tf.keras.layers.Reshape(target_shape=unit)
    elif block == "DROPOUT":
        layer = tf.keras.layers.Dropout(unit)
    elif block == "SPATIAL_DROPOUT":
        layer = tf.keras.layers.SpatialDropout1D(unit)
    elif block == "GLOBAL_MAX_POOL":
        layer = tf.keras.layers.GlobalMaxPooling1D()
    elif block == "MAX_POOL":
        layer = tf.keras.layers.MaxPool1D(pool_size=unit)
    elif block == "GLOBAL_AVERAGE_POOL":
        layer = tf.keras.layers.GlobalAveragePooling1D()
    elif block == "AVERAGE_POOL":
        layer = tf.keras.layers.AveragePooling1D(pool_size=unit)

    return layer
