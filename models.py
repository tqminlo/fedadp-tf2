from keras.layers import *
from keras.models import Model
import tensorflow as tf


def CNN_MNIST(model_name):
    inp = Input(shape=(28, 28, 1))
    x = Conv2D(filters=32, kernel_size=5, strides=1, padding="valid")(inp)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(filters=64, kernel_size=5, strides=1, padding="valid")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(inp, x, name=model_name)

    return model


if __name__ == "__main__":
    model = CNN_MNIST("node0")
    model.summary()
