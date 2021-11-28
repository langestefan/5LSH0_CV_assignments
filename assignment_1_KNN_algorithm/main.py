"""
main.py
https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import euclidean


# import iris dataset

def calculate_distances(data, point):
    """
    Calculate for given point the distance to the other N-1 samples
    :param data: x/y data
    :param point: integer, sample [1-N]
    :return: distance from point to the other N-1 points
    """
    x0, y0 = data[point, 0], data[point, 1]
    x1, y1 = data[point+1, 0], data[point+1, 1]
    print("Point x0/y0 = {0}:{1}".format(x0, y0))
    print("Point x1/y1 = {0}:{1}".format(x1, y1))
    distance = euclidean([x0, y0], [x1, y1])
    print(distance)


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


def app_main():
    """ App main """
    iris = datasets.load_iris()
    data = iris.data[:, :2]  # take first 2 features (sepal length/width)
    label = iris.target

    # visualize_data(data, label)

    calculate_distances(data, 1)


if __name__ == "__main__":
    app_main()
