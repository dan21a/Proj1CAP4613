from convolutional import LocallyConnectedCNN
from fully_connected import FullyConnectedNN
from locally_connected import LocallyConnectedNN
import torch.nn as nn
from train import train
from zip_dataset import ZipDataset

def fully_connected(train_data: ZipDataset, test_data: ZipDataset) -> None:
    # Learning is very slow
    model = FullyConnectedNN()
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        nn.init.uniform_(layer.weight, -0.01, 0.01)
    train(model, train_data, test_data, 32, 0.01).plot()

    # Learning is effective (fast and accurate)
    model = FullyConnectedNN()
    for layer in [model.layer1, model.layer2, model.layer4]:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    nn.init.xavier_uniform_(model.layer3.weight)
    train(model, train_data, test_data, 32, 0.01).plot()

    # Learning is very fast (not accurate)
    model = FullyConnectedNN()
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        nn.init.uniform_(layer.weight, -1, 1)
    train(model, train_data, test_data, 32, 0.01).plot()


def convolutional(train_data: ZipDataset, test_data: ZipDataset) -> None:
    # Learning is very slow
    model = LocallyConnectedCNN()
    for layer in [model.conv1, model.conv2, model.conv3, model.fc1]:
        nn.init.uniform_(layer.weight, -0.000001, 0.000001)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

    # Learning is effective (fast and accurate)
    model = LocallyConnectedCNN()
    for layer in [model.conv1, model.conv2, model.conv3]:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.xavier_normal_(model.fc1.weight)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

    # Learning is very fast (not accurate)
    model = LocallyConnectedCNN()
    for layer in [model.conv1, model.conv2, model.conv3, model.fc1]:
        nn.init.uniform_(layer.weight, -1, 1)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

def locally_connected_training(train_data: ZipDataset, test_data: ZipDataset) -> None:
    # Slow Learning (very slow convergence)
    model = LocallyConnectedNN()
    for layer in [model.local1, model.local2, model.local3]:
        nn.init.uniform_(layer.weights, -0.01, 0.01)
    train(model, train_data, test_data, batch_size=512, learning_rate=0.001, epochs=150).plot()

    # Good Learning (balanced speed and accuracy)
    model = LocallyConnectedNN()
    for layer in [model.local1, model.local2]:
        nn.init.kaiming_uniform_(layer.weights, nonlinearity="relu")
    nn.init.xavier_uniform_(model.local3.weights)
    train(model, train_data, test_data, batch_size=512, learning_rate=0.005, epochs=150).plot()

    #Too Fast Learning (unstable updates, poor accuracy)
    model = LocallyConnectedNN()
    for layer in [model.local1, model.local2, model.local3]:
        nn.init.uniform_(layer.weights, -1, 1)
    train(model, train_data, test_data, batch_size=512, learning_rate=0.01, epochs=150).plot()


if __name__ == "__main__":
    train_data = ZipDataset("../zip_train.txt")
    test_data = ZipDataset("../zip_test.txt")

    fully_connected(train_data, test_data)
    convolutional(train_data, test_data)
    locally_connected_training(train_data, test_data)

