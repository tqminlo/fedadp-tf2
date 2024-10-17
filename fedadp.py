import os
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras.optimizers import SGD, Adam
from models import CNN_MNIST
import numpy as np
import random
import keras
from tqdm import tqdm


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


class ServerAdp:
    def __init__(self, server_w, eval_dir):
        self.server_w = server_w
        self.model = CNN_MNIST("server")
        self.model.compile(loss="categorical_crossentropy", metrics=["acc"])
        self.model.save_weights(self.server_w)
        self.X_test = np.load(os.path.join(eval_dir, "X_test.npy"))
        self.Y_test = np.load(os.path.join(eval_dir, "Y_test.npy"))
        self.X_test = np.expand_dims(self.X_test, axis=-1) / 255.
        self.X_test = (self.X_test - 0.1307) / 0.3081           # norm
        self.Y_test = keras.utils.to_categorical(self.Y_test, num_classes=10)

    def aggregation(self, clients_num_samples, clients_w):
        D = np.array(clients_num_samples)
        D = D / np.sum(D)
        for layer in self.model.layers:
            if "cv" in layer.name or "den" in layer.name:
                clients_w_this_layer = clients_w[layer.name]
                client_kernels = [clients_w_this_layer[i][0] for i in range(10)]
                client_biases = [clients_w_this_layer[i][1] for i in range(10)]
                kernel = [client_kernels[i] * D[i] for i in range(10)]
                bias = [client_biases[i] * D[i] for i in range(10)]
                kernel = sum(kernel)
                bias = sum(bias)
                layer.set_weights([kernel, bias])

        self.model.save_weights(self.server_w)

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
            print(f"---- Round {i}, lr: {self.clients.client_models[0].optimizer.lr.numpy()}")
            self.clients.train_all_members("saved/server_w_0.h5", self.dataset_dir)
            self.server.aggregation(self.clients.num_samples, self.clients.clients_w)
            self.server.eval()

            # for model in self.clients.client_models:
            #     model.optimizer.lr = model.optimizer.lr * 0.995
            #     model.compile(SGD(learning_rate=model.optimizer.lr), loss="categorical_crossentropy", metrics=["acc"])


if __name__ == "__main__":
    fed_adp = FedAdp(300, 32, 1, 0.01, "5i5ni2c_r1")
    fed_adp.pipline()