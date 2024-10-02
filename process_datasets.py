import keras
import numpy as np
from sklearn.utils import shuffle

mnist = keras.datasets.mnist.load_data()
print(mnist[0][0].shape)


def take_test_mnist():
    X_test = mnist[1][0]
    Y_test = mnist[1][1]
    np.save("datasets/mnist/test/X_test.npy", X_test)
    np.save("datasets/mnist/test/Y_test.npy", Y_test)


def take_iid_mnist():
    X_train = mnist[0][0]
    Y_train = mnist[0][1]
    X_train, Y_train = shuffle((X_train, Y_train))
    for i in range(100):
        X_train_node = X_train[i*600: (i+1)*600]
        Y_train_node = Y_train[i*600: (i+1)*600]
        np.save(f"datasets/mnist/iid/X_train_node{i:03}.npy", X_train_node)
        np.save(f"datasets/mnist/iid/Y_train_node{i:03}.npy", Y_train_node)


def take_adp_exp_mnist(num_iid):
    num_niid = 10 - num_iid
    X_train = mnist[0][0]
    Y_train = mnist[0][1]
    X_train, Y_train = shuffle((X_train, Y_train))
    for i in range(num_iid):
        X_train_node = X_train[i * 600: (i + 1) * 600]
        Y_train_node = Y_train[i * 600: (i + 1) * 600]
        np.save(f"datasets/mnist/iid/X_train_iid{i:03}.npy", X_train_node)
        np.save(f"datasets/mnist/iid/Y_train_iid{i:03}.npy", Y_train_node)
    ####################################################


if __name__ == "__main__":
    # take_test_mnist()
    take_iid_mnist()