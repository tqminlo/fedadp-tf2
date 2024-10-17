import keras
import numpy as np
from sklearn.utils import shuffle
import os

mnist = keras.datasets.mnist.load_data()


def take_test_mnist():
    X_test = mnist[1][0]
    Y_test = mnist[1][1]
    np.save("datasets/mnist/test/X_test.npy", X_test)
    np.save("datasets/mnist/test/Y_test.npy", Y_test)


def take_iid_mnist():
    X_train = mnist[0][0]
    Y_train = mnist[0][1]
    X_train, Y_train = shuffle(X_train, Y_train)
    for i in range(100):
        X_train_node = X_train[i*600: (i+1)*600]
        Y_train_node = Y_train[i*600: (i+1)*600]
        np.save(f"datasets/mnist/iid/X_train_node{i:03}.npy", X_train_node)
        np.save(f"datasets/mnist/iid/Y_train_node{i:03}.npy", Y_train_node)


def take_niid_shard_mnist():
    X_train = mnist[0][0]
    Y_train = mnist[0][1]
    X_train, Y_train = shuffle(X_train, Y_train)
    idx_order = np.argsort(Y_train)[::1]
    Y_train = Y_train[idx_order]
    X_train = X_train[idx_order]
    X_train = np.concatenate([X_train[100:], X_train[:100]], axis=0)
    Y_train = np.concatenate([Y_train[100:], Y_train[:100]], axis=0)
    # print(idx_order.tolist())
    idx_shard = np.arange(100)
    idx_shard = np.concatenate([idx_shard, idx_shard], axis=-1)
    # print(idx_shard.shape)
    idx_shard = shuffle(idx_shard)
    for i in range(100):
        node_shard = idx_shard[i*2: (i+1)*2]
        X_train_node = [X_train[node_shard[0]*300: (node_shard[0]+1)*300], X_train[node_shard[1]*300: (node_shard[1]+1)*300]]
        X_train_node = np.concatenate(X_train_node, axis=0)
        Y_train_node = [Y_train[node_shard[0]*300: (node_shard[0]+1)*300], Y_train[node_shard[1]*300: (node_shard[1]+1)*300]]
        Y_train_node = np.concatenate(Y_train_node, axis=0)
        print(X_train_node.shape)
        print(Y_train_node.shape)
        np.save(f"datasets/mnist/niid_shard/X_train_node{i:03}.npy", X_train_node)
        np.save(f"datasets/mnist/niid_shard/Y_train_node{i:03}.npy", Y_train_node)


def take_adp_exp_mnist(num_iid=5, x_class=2, dir_name="5iid5niid_2class"):
    os.makedirs(f"datasets/mnist/{dir_name}", exist_ok=True)
    num_niid = 10 - num_iid
    X_train = mnist[0][0]
    Y_train = mnist[0][1]
    X_train, Y_train = shuffle(X_train, Y_train)

    for i in range(num_iid):
        X_train_node = X_train[600 * i: 600 * (i + 1)]
        Y_train_node = Y_train[600 * i: 600 * (i + 1)]

        check_label = [0] * 10
        for l in Y_train_node:
            check_label[l] += 1
        print(i, check_label)

        np.save(f"datasets/mnist/{dir_name}/X_train_node{i:03}.npy", X_train_node)
        np.save(f"datasets/mnist/{dir_name}/Y_train_node{i:03}.npy", Y_train_node)

    start_id = 600 * num_iid
    for i in range(num_iid, 10):
        all_id = shuffle(np.arange(10))
        client_class = all_id[:x_class]
        X_train_node, Y_train_node = [], []
        num_data_client = 0
        for j in range(start_id, 60000):
            if Y_train[j] in client_class:
                X_train_node.append(X_train[j])
                Y_train_node.append(Y_train[j])
                num_data_client += 1
            if num_data_client == 600:
                start_id = j
                break

        check_label = [0] * 10
        for l in Y_train_node:
            check_label[l] += 1
        print(i, check_label)

        X_train_node = np.array(X_train_node)
        Y_train_node = np.array(Y_train_node)
        np.save(f"datasets/mnist/{dir_name}/X_train_node{i:03}.npy", X_train_node)
        np.save(f"datasets/mnist/{dir_name}/Y_train_node{i:03}.npy", Y_train_node)


if __name__ == "__main__":
    # take_test_mnist()
    # take_iid_mnist()
    # take_niid_shard_mnist()
    # take_adp_exp_mnist(10, "10iid")
    take_adp_exp_mnist(num_iid=5, x_class=2, dir_name="5i5ni2c_r1")