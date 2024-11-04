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


class ClientsAvg:
    def __init__(self, num_all_client, ratio_c, batch_size, epochs, lr):
        self.num_all_client = num_all_client
        self.ratio_c = ratio_c
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.num_client_a_round = int(num_all_client * ratio_c)
        self.client_models = [CNN_MNIST(f"member{i:03}") for i in range(self.num_client_a_round)]
        for model in self.client_models:
            model.compile(SGD(learning_rate=lr), loss="categorical_crossentropy", metrics=["acc"])
        self.clients_w = {}
        for layer in self.client_models[0].layers:
            if "cv" in layer.name or "den" in layer.name:
                self.clients_w[layer.name] = [0] * self.num_client_a_round
        self.num_samples = [0] * self.num_client_a_round

    def train_all_members(self, server_w, members_id, dataset_dir):
        for model in self.client_models:
            model.load_weights(server_w)

        pbar = tqdm(range(self.num_client_a_round))
        for i in pbar:
            # print(f"------------- member {i}")
            model = self.client_models[i]
            idx = members_id[i]
            X_train = np.load(os.path.join(dataset_dir, f"X_train_node{idx:03}.npy"))
            Y_train = np.load(os.path.join(dataset_dir, f"Y_train_node{idx:03}.npy"))
            X_train = np.expand_dims(X_train, axis=-1) / 255.
            X_train = (X_train - 0.1307) / 0.3081               # norm
            Y_train = keras.utils.to_categorical(Y_train, num_classes=10)
            self.num_samples[i] = len(X_train)

            model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            for layer in model.layers:
                if "cv" in layer.name or "den" in layer.name:
                    w = layer.get_weights()
                    self.clients_w[layer.name][i] = w

            pbar.set_description(f"Processing client {idx}")


class ServerAvg:
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
        num_participant = len(clients_num_samples)
        for layer in self.model.layers:
            if "cv" in layer.name or "den" in layer.name:
                clients_w_this_layer = clients_w[layer.name]
                client_kernels = [clients_w_this_layer[i][0] for i in range(num_participant)]
                client_biases = [clients_w_this_layer[i][1] for i in range(num_participant)]
                kernel = [client_kernels[i] * D[i] for i in range(num_participant)]
                bias = [client_biases[i] * D[i] for i in range(num_participant)]
                kernel = sum(kernel)
                bias = sum(bias)
                layer.set_weights([kernel, bias])

        self.model.save_weights(self.server_w)

    def eval(self):
        return self.model.evaluate(self.X_test, self.Y_test)


class FedAvg:
    def __init__(self, num_round, num_all_client, ratio_c, batch_size, epochs, lr, distribution="iid"):
        self.num_round = num_round
        self.dataset_dir = f"datasets/mnist/{distribution}"
        self.clients = ClientsAvg(num_all_client, ratio_c, batch_size, epochs, lr)
        self.server = ServerAvg("saved/server_w.h5", "datasets/mnist/test")
        self.num_client_a_round = self.clients.num_client_a_round
        self.num_all_client = self.clients.num_all_client

    def pipline(self):
        all_id = [i for i in range(self.num_all_client)]
        loss = 100
        for i in range(self.num_round):
            members_id = random.sample(all_id, self.num_client_a_round)
            print(f"---- Round {i}, lr: {self.clients.client_models[0].optimizer.lr.numpy()}, clients: {members_id}")
            self.clients.train_all_members("saved/server_w.h5", members_id, self.dataset_dir)
            self.server.aggregation(self.clients.num_samples, self.clients.clients_w)
            new_loss = self.server.eval()[0]
            for model in self.clients.client_models:
                model.optimizer.lr = model.optimizer.lr * 0.995
                model.compile(SGD(learning_rate=model.optimizer.lr), loss="categorical_crossentropy", metrics=["acc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_round", type=int, default=300)
    parser.add_argument("--all_clients", type=int, default=100)
    parser.add_argument("--ratio_c", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--exp", type=str, default="iid")
    args = parser.parse_args()

    fed_avg = FedAvg(args.num_round, args.all_clients, args.ratio_c, args.batch_size, args.epochs, args.lr, args.exp)
    fed_avg.pipline()