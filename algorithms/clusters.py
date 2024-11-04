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
from fedavg import *
from sklearn.cluster import KMeans, MeanShift


class FedClusters:
    def __init__(self, num_round, num_all_client, ratio_c, batch_size, epochs, lr, distribution="clusters_e0"):
        self.num_round = num_round
        self.dataset_dir = f"datasets/mnist/{distribution}"
        self.distribution = distribution
        self.clients = ClientsAvg(num_all_client, ratio_c, batch_size, epochs, lr)
        self.server = ServerAvg("saved/server_w_1.h5", "datasets/mnist/test")
        self.num_client_a_round = self.clients.num_client_a_round
        self.num_all_client = self.clients.num_all_client
        if distribution == "clusters_e0":
            num_cluster = 5
        else:
            num_cluster = 10
        self.clust = KMeans(n_clusters=num_cluster)
        # self.clust = MeanShift(bandwidth=10)

    def clustering(self):
        check_label_all = []
        for i in range(100):
            label = f"Y_train_node{i:03}.npy"
            label_path = os.path.join(self.dataset_dir, label)
            Y = np.load(label_path)
            check_label = np.zeros(shape=(10,), dtype=int)
            for l in Y:
                check_label[l] += 1
            check_label_all.append(check_label)

        check_label_all = np.array(check_label_all)
        clusters = self.clust.fit(check_label_all)
        print("-------")
        print(clusters.labels_)

        return clusters.__dict__["labels_"]

    def pipline(self):
        all_id = [i for i in range(self.num_all_client)]
        cluster_labels = self.clustering()
        # for i in range(self.num_round):
        #     if self.distribution == "clusters_e0":
        #         members_id = (random.sample(all_id[:20], 2) + random.sample(all_id[20:30], 1) +
        #                       random.sample(all_id[30:50], 2) + random.sample(all_id[50:60], 1) +
        #                       random.sample(all_id[60:], 4))
        #     elif self.distribution == "clusters_e1":
        #         members_id = [random.randint(j*10, j*10+9) for j in range(10)]
        #     else:
        #         members_id = [random.randint(j * 10, j * 10 + 9) for j in range(10)]
        #     print(f"---- Round {i}, lr: {self.clients.client_models[0].optimizer.lr.numpy()}, clients: {members_id}")
        #     self.clients.train_all_members("saved/server_w_1.h5", members_id, self.dataset_dir)
        #     self.server.aggregation(self.clients.num_samples, self.clients.clients_w)
        #     new_loss = self.server.eval()[0]
        #     for model in self.clients.client_models:
        #         model.optimizer.lr = model.optimizer.lr * 0.995
        #         model.compile(SGD(learning_rate=model.optimizer.lr), loss="categorical_crossentropy", metrics=["acc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_round", type=int, default=300)
    parser.add_argument("--all_clients", type=int, default=100)
    parser.add_argument("--ratio_c", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--exp", type=str, default="clusters_e0")
    args = parser.parse_args()

    fed_clusters = FedClusters(args.num_round, args.all_clients, args.ratio_c, args.batch_size, args.epochs, args.lr, args.exp)
    fed_clusters.pipline()
