import keras
import numpy as np
from sklearn.utils import shuffle
import os

mnist = keras.datasets.mnist.load_data()
X_train = mnist[0][0]
Y_train = mnist[0][1]
# check_label = [0] * 10
# for l in Y_train:
#     check_label[l] += 1
# print(check_label)


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


def take_cluster_exp_mnist(case=0, dir_name="clusters_e0"):
    X_train = mnist[0][0]
    Y_train = mnist[0][1]
    X_train, Y_train = shuffle(X_train, Y_train)

    os.makedirs(f"datasets/mnist/{dir_name}", exist_ok=True)

    if case == 0:
        cluster_ids = [0, 1, 2, 3, 4]
        num_in_clusters = [20, 10, 20, 10, 40]
        signs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1], [0, 5], [2, 3], [4, 6]]
        num_in_signs = [[60, 60, 60, 60, 60, 60, 60, 60, 60, 60], [200], [200, 200], [100, 200], [100, 100]]

        # For cluster-0 :
        for i in range(20):
            X_train_node = X_train[i * 600: (i + 1) * 600]
            Y_train_node = Y_train[i * 600: (i + 1) * 600]
            np.save(f"datasets/mnist/{dir_name}/X_train_node{i:03}.npy", X_train_node)
            np.save(f"datasets/mnist/{dir_name}/Y_train_node{i:03}.npy", Y_train_node)

        # For cluster-1to4 :
        X_train = X_train[12000:]
        Y_train = Y_train[12000:]
        idx_order = np.argsort(Y_train)[::1]
        Y_train = Y_train[idx_order].tolist()
        X_train = X_train[idx_order].tolist()
        id_starts = []
        check = 0
        for i in range(len(Y_train)):
            if Y_train[i] == check:
                id_starts.append(i)
                check += 1
        print(id_starts)

        clusters_X = [[] for i in range(5)]
        clusters_Y = [[] for i in range(5)]

        clusters_X[1] += X_train[id_starts[1]: id_starts[1] + 2000]
        clusters_Y[1] += Y_train[id_starts[1]: id_starts[1] + 2000]

        clusters_X[2] += X_train[id_starts[0]: id_starts[0] + 4000]
        clusters_Y[2] += Y_train[id_starts[0]: id_starts[0] + 4000]
        clusters_X[2] += X_train[id_starts[5]: id_starts[5] + 4000]
        clusters_Y[2] += Y_train[id_starts[5]: id_starts[5] + 4000]

        clusters_X[3] += X_train[id_starts[2]: id_starts[2] + 1000]
        clusters_Y[3] += Y_train[id_starts[2]: id_starts[2] + 1000]
        clusters_X[3] += X_train[id_starts[3]: id_starts[3] + 2000]
        clusters_Y[3] += Y_train[id_starts[3]: id_starts[3] + 2000]

        clusters_X[4] += X_train[id_starts[4]: id_starts[4] + 4000]
        clusters_Y[4] += Y_train[id_starts[4]: id_starts[4] + 4000]
        clusters_X[4] += X_train[id_starts[6]: id_starts[6] + 4000]
        clusters_Y[4] += Y_train[id_starts[6]: id_starts[6] + 4000]

        X_residual = (X_train[id_starts[0]+4000 : id_starts[1]] + X_train[id_starts[1]+2000 : id_starts[2]] +
                      X_train[id_starts[2]+1000 : id_starts[3]] + X_train[id_starts[3]+2000 : id_starts[4]] +
                      X_train[id_starts[4]+4000 : id_starts[5]] + X_train[id_starts[5]+4000 : id_starts[6]] +
                      X_train[id_starts[6]+4000 :])
        Y_residual = (Y_train[id_starts[0] + 4000: id_starts[1]] + Y_train[id_starts[1] + 2000: id_starts[2]] +
                      Y_train[id_starts[2] + 1000: id_starts[3]] + Y_train[id_starts[3] + 2000: id_starts[4]] +
                      Y_train[id_starts[4] + 4000: id_starts[5]] + Y_train[id_starts[5] + 4000: id_starts[6]] +
                      Y_train[id_starts[6] + 4000:])

        X_residual, Y_residual = shuffle(X_residual, Y_residual)

        clusters_X[1] += X_residual[:4000]
        clusters_Y[1] += Y_residual[:4000]
        clusters_X[1], clusters_Y[1] = shuffle(clusters_X[1], clusters_Y[1])

        clusters_X[2] += X_residual[4000: 8000]
        clusters_Y[2] += Y_residual[4000: 8000]
        clusters_X[2], clusters_Y[2] = shuffle(clusters_X[2], clusters_Y[2])

        clusters_X[3] += X_residual[8000: 11000]
        clusters_Y[3] += Y_residual[8000: 11000]
        clusters_X[3], clusters_Y[3] = shuffle(clusters_X[3], clusters_Y[3])

        clusters_X[4] += X_residual[11000:]
        clusters_Y[4] += Y_residual[11000:]
        clusters_X[4], clusters_Y[4] = shuffle(clusters_X[4], clusters_Y[4])

        for i in range(10):
            X_train_node = np.array(clusters_X[1][i * 600: (i + 1) * 600])
            Y_train_node = np.array(clusters_Y[1][i * 600: (i + 1) * 600])
            np.save(f"datasets/mnist/{dir_name}/X_train_node{i+20:03}.npy", X_train_node)
            np.save(f"datasets/mnist/{dir_name}/Y_train_node{i+20:03}.npy", Y_train_node)

        for i in range(20):
            X_train_node = np.array(clusters_X[2][i * 600: (i + 1) * 600])
            Y_train_node = np.array(clusters_Y[2][i * 600: (i + 1) * 600])
            np.save(f"datasets/mnist/{dir_name}/X_train_node{i+30:03}.npy", X_train_node)
            np.save(f"datasets/mnist/{dir_name}/Y_train_node{i+30:03}.npy", Y_train_node)

        for i in range(10):
            X_train_node = np.array(clusters_X[3][i * 600: (i + 1) * 600])
            Y_train_node = np.array(clusters_Y[3][i * 600: (i + 1) * 600])
            np.save(f"datasets/mnist/{dir_name}/X_train_node{i+50:03}.npy", X_train_node)
            np.save(f"datasets/mnist/{dir_name}/Y_train_node{i+50:03}.npy", Y_train_node)

        for i in range(40):
            X_train_node = np.array(clusters_X[4][i * 600: (i + 1) * 600])
            Y_train_node = np.array(clusters_Y[4][i * 600: (i + 1) * 600])
            np.save(f"datasets/mnist/{dir_name}/X_train_node{i+60:03}.npy", X_train_node)
            np.save(f"datasets/mnist/{dir_name}/Y_train_node{i+60:03}.npy", Y_train_node)

    if case == 1:
        cluster_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_in_clusters = [10] * 10
        signs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_in_signs = [400] * 10

        idx_order = np.argsort(Y_train)[::1]
        Y_train = Y_train[idx_order].tolist()
        X_train = X_train[idx_order].tolist()
        id_starts = []
        check = 0
        for i in range(len(Y_train)):
            if Y_train[i] == check:
                id_starts.append(i)
                check += 1
        print(id_starts)

        clusters_X = [[] for i in range(10)]
        clusters_Y = [[] for i in range(10)]
        X_residual = []
        Y_residual = []
        for i in range(10):
            clusters_X[i] += X_train[id_starts[i]: id_starts[i] + 4000]
            clusters_Y[i] += Y_train[id_starts[i]: id_starts[i] + 4000]
            X_residual += X_train[id_starts[i] + 4000: id_starts[i+1]] if i < 9 else X_train[id_starts[i] + 4000:]
            Y_residual += Y_train[id_starts[i] + 4000: id_starts[i+1]] if i < 9 else Y_train[id_starts[i] + 4000:]

        X_residual, Y_residual = shuffle(X_residual, Y_residual)
        for i in range(10):                 # 10 clusters
            clusters_X[i] += X_residual[:2000]
            clusters_Y[i] += Y_residual[:2000]
            clusters_X[i], clusters_Y[i] = shuffle(clusters_X[i], clusters_Y[i])

            for j in range(10):                 # 10 clients in cluster
                X_train_node = np.array(clusters_X[i][j * 600: (j + 1) * 600])
                Y_train_node = np.array(clusters_Y[i][j * 600: (j + 1) * 600])
                np.save(f"datasets/mnist/{dir_name}/X_train_node{(i*10+j):03}.npy", X_train_node)
                np.save(f"datasets/mnist/{dir_name}/Y_train_node{(i*10+j):03}.npy", Y_train_node)

    if case == 2:
        cluster_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_in_clusters = [10] * 10
        signs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_in_signs = [600] * 10

        idx_order = np.argsort(Y_train)[::1]
        Y_train = Y_train[idx_order].tolist()
        X_train = X_train[idx_order].tolist()
        id_starts = []
        check = 0
        for i in range(len(Y_train)):
            if Y_train[i] == check:
                id_starts.append(i)
                check += 1
        print(id_starts)

        clusters_X = [[] for i in range(10)]
        clusters_Y = [[] for i in range(10)]
        for i in range(10):
            clusters_X[i] = X_train[i*6000: (i+1)*6000]
            clusters_Y[i] = Y_train[i*6000: (i+1)*6000]
            clusters_X[i], clusters_Y[i] = shuffle(clusters_X[i], clusters_Y[i])

            for j in range(10):  # 10 clients in cluster
                X_train_node = np.array(clusters_X[i][j * 600: (j + 1) * 600])
                Y_train_node = np.array(clusters_Y[i][j * 600: (j + 1) * 600])
                np.save(f"datasets/mnist/{dir_name}/X_train_node{(i * 10 + j):03}.npy", X_train_node)
                np.save(f"datasets/mnist/{dir_name}/Y_train_node{(i * 10 + j):03}.npy", Y_train_node)

    if case == 3:
        cluster_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_in_clusters = [10] * 10
        signs = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [0,4,9], [1,6,9], [1,5,8], [2,3,7], [2,4,9], [5,7,8]]
        num_in_signs = [400] * 10


if __name__ == "__main__":
    # take_test_mnist()
    # take_iid_mnist()
    # take_niid_shard_mnist()
    # take_adp_exp_mnist(10, "10iid")
    # take_adp_exp_mnist(num_iid=5, x_class=2, dir_name="5i5ni2c_r1")
    # take_cluster_exp_mnist(dir_name="clusters_e0")
    take_cluster_exp_mnist(case=2, dir_name="clusters_e2")