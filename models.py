from keras.layers import *
from keras.models import Model
import tensorflow as tf


def CNN_MNIST(model_name):
    inp = Input(shape=(28, 28, 1))
    x = ZeroPadding2D(padding=1)(inp)
    x = Conv2D(filters=32, kernel_size=5, strides=1, name="cv0")(x)
    x = ZeroPadding2D(padding=1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = ZeroPadding2D(padding=1)(x)
    x = Conv2D(filters=64, kernel_size=5, strides=1, name="cv1")(x)
    x = ZeroPadding2D(padding=1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu", name="den0")(x)
    x = Dense(10, activation="softmax", name="den1")(x)

    model = Model(inp, x, name=model_name)

    return model


if __name__ == "__main__":
    model = CNN_MNIST("node0")
    model.summary()