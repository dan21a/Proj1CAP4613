from convolutional import LocallyConnectedCNN
from fully_connected import FullyConnectedNN
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
    # model = LocallyConnectedCNN()
    # for layer in [model.conv1, model.conv2, model.conv3]:
    #     nn.init.uniform_(layer.weight, -0.000001, 0.000001)
    # train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

    # Learning is very fast (not accurate)



if __name__ == "__main__":
    train_data = ZipDataset("../zip_train.txt")
    test_data = ZipDataset("../zip_test.txt")

    # fully_connected(train_data, test_data)
    convolutional(train_data, test_data)

