import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras


def buid_mlp():
    Seq_model = keras.models.Sequential()
    Seq_model.add(keras.layers.Flatten(input_shape=[28,28]))
    Seq_model.add(keras.layers.Dense(300, activation="relu"))
    Seq_model.add(keras.layers.Dense(100, activation="relu"))
    Seq_model.add(keras.layers.Dense(10, activation="softmax"))
    
    Seq_model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.sparse_categorical_accuracy])
    
    return Seq_model
