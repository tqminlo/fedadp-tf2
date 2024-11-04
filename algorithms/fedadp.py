import os
import sys
sys.path.append("E:/tqminlo\Master\FL/fedadp-tf2")
from keras.optimizers import SGD, Adam
from models import CNN_MNIST
import numpy as np
import random
import keras
from tqdm import tqdm
import argparse
from fedavg import ServerAvg


class ClientsAdp:
    def __init__(self, batch_size, epochs, lr):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.client_models = [CNN_MNIST(f"member{i:03}") for i in range(10)]
        for model in self.client_models:
            model.compile(SGD(learning_rate=lr), loss="categorical_crossentropy", metrics=["acc"])
        self.clients_w = {}
        for layer in self.client_models[0].layers:
            if "cv" in layer.name or "den" in layer.name:
                self.clients_w[layer.name] = [0] * 10
        self.num_samples = [0] * 10

    def train_all_members(self, server_w, dataset_dir):
        for model in self.client_models:
            model.load_weights(server_w)

        pbar = tqdm(range(10))
        for i in pbar:
            # print(f"------------- member {i}")
            model = self.client_models[i]
            X_train = np.load(os.path.join(dataset_dir, f"X_train_node{i:03}.npy"))
            Y_train = np.load(os.path.join(dataset_dir, f"Y_train_node{i:03}.npy"))
            X_train = np.expand_dims(X_train, axis=-1) / 255.
            X_train = (X_train - 0.1307) / 0.3081               # norm
            Y_train = keras.utils.to_categorical(Y_train, num_classes=10)
            self.num_samples[i] = len(X_train)

            model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            for layer in model.layers:
                if "cv" in layer.name or "den" in layer.name:
                    w = layer.get_weights()
                    self.clients_w[layer.name][i] = w

            pbar.set_description(f"Processing client {i}")


class ServerAdp(ServerAvg):
    def __init__(self, server_w, eval_dir):
        super().__init__(server_w, eval_dir)
        self.pre_w = []
        for layer in self.model.layers:
            if "cv" in layer.name or "den" in layer.name:
                w = layer.get_weights()
                kernel = w[0].flatten()
                bias = w[1].flatten()
                self.pre_w.append(kernel)
                self.pre_w.append(bias)
        self.pre_w = np.concatenate(self.pre_w, axis=0)
        self.pre_thetas = None

    def cosine_similarity(self, a, b):
        return np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

    def gompertz_func(self, theta, alpha=5.):
        return alpha * (1 - np.exp(-np.exp(-alpha * (theta - 1))))

    def aggregation(self, clients_num_samples, clients_w, t):
        D = np.array(clients_num_samples)
        D = D / np.sum(D)
        num_participant = len(clients_num_samples)  # = 10

        flatten_ws = [[] for i in range(num_participant + 1)]  # 10 clients + 1 global
        for layer in self.model.layers:
            if "cv" in layer.name or "den" in layer.name:
                clients_w_this_layer = clients_w[layer.name]
                client_kernels = [clients_w_this_layer[i][0].flatten() for i in range(num_participant)]
                client_biases = [clients_w_this_layer[i][1].flatten() for i in range(num_participant)]

                for i in range(num_participant):
                    flatten_ws[i].append(client_kernels[i])
                    flatten_ws[i].append(client_biases[i])

                global_kernel = [client_kernels[i] * D[i] for i in range(num_participant)]
                global_bias = [client_biases[i] * D[i] for i in range(num_participant)]
                global_kernel = sum(global_kernel)
                global_bias = sum(global_bias)

                flatten_ws[-1].append(global_kernel)
                flatten_ws[-1].append(global_bias)

        flatten_ws = [np.concatenate(w, axis=0) for w in flatten_ws]

        gradients = [self.pre_w - w for w in flatten_ws]
        client_gs = gradients[:-1]
        global_g = gradients[-1]
        cosine_similarities = [self.cosine_similarity(g, global_g) for g in client_gs]
        thetas = np.array([np.arccos(cos) for cos in cosine_similarities])
        print("----1---- thetas:", thetas)
        print("----2---- pre_thetas:", self.pre_thetas)

        if self.pre_thetas is None:
            thetas_smooth = thetas
        else:
            thetas_smooth = self.pre_thetas * (t-1)/t + thetas * 1/t
        print("----3---- thetas_smooth:", thetas_smooth)

        re_weights = np.exp(self.gompertz_func(thetas_smooth, alpha=5.))
        re_weights = re_weights * D
        re_weights = re_weights / np.sum(re_weights)
        print("----4---- re_weights:", re_weights)

        assert np.all(re_weights >= 0), "weights should be non-negative values"
        new_w = [flatten_ws[i] * re_weights[i] for i in range(num_participant)]
        new_w = sum(new_w)
        for layer in self.model.layers:
            if "cv" in layer.name or "den" in layer.name:
                clients_w_this_layer = clients_w[layer.name]
                client_kernels = [clients_w_this_layer[i][0] for i in range(num_participant)]
                client_biases = [clients_w_this_layer[i][1] for i in range(num_participant)]
                kernel = [client_kernels[i] * re_weights[i] for i in range(num_participant)]
                bias = [client_biases[i] * re_weights[i] for i in range(num_participant)]
                kernel = sum(kernel)
                bias = sum(bias)
                layer.set_weights([kernel, bias])

        self.model.save_weights(self.server_w)
        self.pre_w = new_w
        self.pre_thetas = thetas_smooth

    def eval(self):
        return self.model.evaluate(self.X_test, self.Y_test)


class FedAdp:
    def __init__(self, num_round, batch_size, epochs, lr, distribution="iid"):
        self.num_round = num_round
        self.dataset_dir = f"datasets/mnist/{distribution}"
        self.clients = ClientsAdp(batch_size, epochs, lr)
        self.server = ServerAdp("saved/server_w_0.h5", "datasets/mnist/test")

    def pipline(self):
        for i in range(self.num_round):
            print(f"---- Round {i+1}, lr: {self.clients.client_models[0].optimizer.lr.numpy()}")
            self.clients.train_all_members("saved/server_w_0.h5", self.dataset_dir)
            self.server.aggregation(self.clients.num_samples, self.clients.clients_w, i+1)
            self.server.eval()

            # for model in self.clients.client_models:
            #     model.optimizer.lr = model.optimizer.lr * 0.995
            #     model.compile(SGD(learning_rate=model.optimizer.lr), loss="categorical_crossentropy", metrics=["acc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_round", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--exp", type=str, default="5i5ni2c_r1")
    args = parser.parse_args()

    fed_adp = FedAdp(args.num_round, args.batch_size, args.epochs, args.lr, args.exp)
    fed_adp.pipline()