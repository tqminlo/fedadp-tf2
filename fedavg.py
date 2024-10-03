import os
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras.optimizers import SGD
from models import CNN_MNIST
import numpy as np


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
            model.compile(SGD(learning_rate=lr, weight_decay=None), loss="sparse_categorical_crossentropy", metrics=["acc"])

    def train_all_members(self, server_w, members_id, dataset_dir):
        for model in self.client_models:
            model.load_weights(server_w)
        for i in range(self.num_client_a_round):
            model = self.client_models[i]
            idx = members_id[i]
            X_train = np.load(os.path.join(dataset_dir, f"X_train_node{idx:03}.npy"))
            Y_train = np.load(os.path.join(dataset_dir, f"X_train_node{idx:03}.npy"))
            model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size)






def client_train(client_id, model, data):
    """
    :param client_id:
    :param model:
    :param data: data = [X_train, Y_train]
    :return:
    """
    model.fit(data[0], data[1])

def server_aggregation():
    a=0
    return


def all_a_round():
    return


if __name__ == "__main__":
    a = 1