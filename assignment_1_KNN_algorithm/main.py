"""
main.py
"""

from sklearn.datasets import load_iris


def app_main():
    """ App main """
    print('test')
    data = load_iris()
    data.target[[10, 25, 50]]
    print(list(data.target_names))


if __name__ == "__main__":
    app_main()
