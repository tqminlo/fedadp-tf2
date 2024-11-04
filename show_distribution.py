import os.path
import random
import numpy as np
import matplotlib.pyplot as plt

check_label_all = []
for i in range(100):
    label = f"Y_train_node{i:03}.npy"
    label_path = os.path.join("datasets/mnist/clusters_e0", label)
    Y = np.load(label_path)
    check_label = [0] * 10
    for l in Y:
        check_label[l] += 1
    check_label_all.append(check_label)
print(check_label_all)


X = [i for i in range(100)]
Ys = [[] for i in range(10)]
for i in range(10):
    Ys[i] = [check_label_all[j][i] for j in range(100)]
    Ys[i] = np.array(Ys[i])
print(Ys)

colors = ["red", "blue", "yellow", "green", "cyan", "tan", "orange", "purple", "pink", "olive"]
start = np.zeros(shape=(100,), dtype=int)
for i in range(10):
    plt.bar(X, Ys[i], bottom=start, color=colors[i])
    start += Ys[i]

plt.xlabel("Client")
plt.ylabel("Num samples")
plt.legend(["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"])
plt.title("Distribution")
plt.show()