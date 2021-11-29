"""
main.py
https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from scipy import stats
import numpy as np


def calculate_distances(data, label, point):
    """
    Calculate for given point the euclidean distance to the other N-1 samples
    :param label: target type of flower
    :param data: x/y data
    :param point: integer, sample [1-N]
    :return: distance from point to the other N-1 points
    """
    size = np.size(data, 0)
    x0, y0 = point[0], point[1]
    distance = np.zeros((size, 2))

    for idx, sample in enumerate(data):
        dis = euclidean([x0, y0], [data[idx, 0], data[idx, 1]])
        distance[idx][0] = label[idx]
        distance[idx][1] = dis

    return distance


def visualize_data(data, label):
    """
    :param data: x/y data
    :param label: labels
    """
    x_min = data[:, 0].min() - 0.5
    x_max = data[:, 0].max() + 0.5
    y_min = data[:, 1].min() - 0.5
    y_max = data[:, 1].max() + 0.5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Set1, edgecolor="black")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("red=setosa, orange=versicolor, grey=virginica")

    plt.show()


def find_k_nearest_neighbors(distances, k):
    """
    :param k: How many neighbor points K we wish to include
    :param distances: distance from point to the other N-1 points
    :return Target labels of K nearest neighbors (K-NN)
    """
    distances = distances[distances[:, 1].argsort()]
    k_nn = np.delete(distances[0:k], 1, 1)
    return k_nn


def app_main():
    """ App main """
    iris = datasets.load_iris()
    data = iris.data[:, :2]  # take first 2 features (sepal length/width)
    label = iris.target
    k = 5  # the K in K-NN
    n_correct = 0

    # split into train test sets
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    n_test = np.size(x_test, 0)

    # visualize_data(data, label)
    # visualize_data(x_train, y_train)
    # visualize_data(x_test, y_test)

    for idx, testp in enumerate(x_test):
        distances = calculate_distances(x_train, y_train, testp)
        k_nn = find_k_nearest_neighbors(distances, k)
        k_mode = stats.mode(k_nn, axis=None)
        # print("Correct label: {0}".format(int(y_test[idx])))
        # print("Majority vote: {0}".format(int(k_mode.mode)))
        if int(y_test[idx]) == int(k_mode.mode):
            n_correct += 1

    accuracy = 100 * n_correct / n_test

    print("Accuracy: {0}%".format(accuracy))


if __name__ == "__main__":
    app_main()

# answers
# Vary the train and test splits. What do you observe and why does it happen?
# -> For high number of test samples the accuracy goes down. This happens because there is not enough data to train the
# model.

# What happens when k=1?
# -> Accuracy varies a lot. There is only 1 direct neighbor to compare against so it's quite random.

# What do you observe when k > 1?
# -> Model is pretty accurate (~80%)

# Can you think of a way to improve the selection of the neighbours?
# --> We can use a weights system and train the weights
