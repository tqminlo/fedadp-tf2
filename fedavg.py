from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras.optimizers import SGD
from models import CNN_MNIST


class ClientAvg:
    def __init__(self, num_all_client, ratio_c, batch_size, epochs, lr):
        self.num_all_client = num_all_client
        self.ratio_c = ratio_c
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.num_client_a_round = int(num_all_client * ratio_c)
        self.client_models = [CNN_MNIST(f"client{i:03}") for i in range(self.num_client_a_round)]
        for model in self.client_models:
            model.compile(SGD(learning_rate=lr, weight_decay=0.995), loss="categorical_crossentropy", metrics=["acc"])


    # def train(self):


def client_train(client_id, model, data):
    """
    :param client_id:
    :param model:
    :param data: data = [X_train, Y_train]
    :return:
    """
    model.fit(data[0], data[1])

def server_aggregation():
    return


def all_a_round():
    return


if __name__ == "__main__":
    a = 1