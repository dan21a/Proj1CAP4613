from convolutional import LocallyConnectedCNN
from fully_connected import FullyConnectedNN
from locally_connected import LocallyConnectedNN
import torch.nn as nn
from train import train
from zip_dataset import ZipDataset

def fully_connected(train_data: ZipDataset, test_data: ZipDataset) -> None:

    model = FullyConnectedNN()
    for layer in [model.layer1, model.layer2, model.layer4]:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    nn.init.xavier_uniform_(model.layer3.weight)
    train(model, train_data, test_data, 32, 0.00001).plot()

    model = FullyConnectedNN()
    for layer in [model.layer1, model.layer2, model.layer4]:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    nn.init.xavier_uniform_(model.layer3.weight)
    train(model, train_data, test_data, 32, 0.01).plot()

    model = FullyConnectedNN()
    for layer in [model.layer1, model.layer2, model.layer4]:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    nn.init.xavier_uniform_(model.layer3.weight)
    train(model, train_data, test_data, 32, 0.1).plot()



def convolutional(train_data: ZipDataset, test_data: ZipDataset) -> None:
    
    model = LocallyConnectedCNN()
    for layer in [model.conv1, model.conv2, model.conv3]:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.xavier_normal_(model.fc1.weight)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.00001, epochs=150).plot()

    model = LocallyConnectedCNN()
    for layer in [model.conv1, model.conv2, model.conv3]:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.xavier_normal_(model.fc1.weight)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

    model = LocallyConnectedCNN()
    for layer in [model.conv1, model.conv2, model.conv3]:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.xavier_normal_(model.fc1.weight)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.1, epochs=150).plot()


def locally_connected_training(train_data: ZipDataset, test_data: ZipDataset) -> None:

    model = LocallyConnectedNN()
    for layer in [model.local1, model.local2]:
        nn.init.kaiming_uniform_(layer.weights, nonlinearity="relu")
    nn.init.xavier_uniform_(model.local3.weights)
    train(model, train_data, test_data, batch_size=512, learning_rate=0.00001, epochs=150).plot()

    model = LocallyConnectedNN()
    for layer in [model.local1, model.local2]:
        nn.init.kaiming_uniform_(layer.weights, nonlinearity="relu")
    nn.init.xavier_uniform_(model.local3.weights)
    train(model, train_data, test_data, batch_size=512, learning_rate=0.005, epochs=150).plot()

    model = LocallyConnectedNN()
    for layer in [model.local1, model.local2]:
        nn.init.kaiming_uniform_(layer.weights, nonlinearity="relu")
    nn.init.xavier_uniform_(model.local3.weights)
    train(model, train_data, test_data, batch_size=512, learning_rate=0.1, epochs=150).plot()


if __name__ == "__main__":
    train_data = ZipDataset("../zip_train.txt")
    test_data = ZipDataset("../zip_test.txt")

    # fully_connected(train_data, test_data)
    convolutional(train_data, test_data)
    # locally_connected_training(train_data, test_data)

