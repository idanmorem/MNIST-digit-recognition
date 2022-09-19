from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


def plot_loss(network):
    epochs = range(0, network.epochs)
    plt.plot(epochs, network.train_loss)
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.figtext(0.2, 0.025, f"The score is:{network.test_score}%", ha="center", va="center", fontsize=18,
                bbox={"facecolor": "white", "alpha": 0.5})
    plt.savefig("Training_loss.png")


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 6000)
    x_test, y_test = preprocess_data(x_test, y_test, 1000)
    network = NeuralNetwork()
    network.fit(x_train, y_train)
    network.score(x_test, y_test)
    plot_loss(network)
