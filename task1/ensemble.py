import matplotlib.pyplot as plt
from convolutional import LocallyConnectedCNN
from fully_connected import FullyConnectedNN
from locally_connected import LocallyConnectedNN
import torch.nn as nn
from train import train, get_gpu, trainEnsemble
from zip_dataset import ZipDataset

def fully_connected(train_data: ZipDataset, test_data: ZipDataset) -> None:
    model = FullyConnectedNN().to(device)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

def convolutional(train_data: ZipDataset, test_data: ZipDataset) -> None:
    model = LocallyConnectedCNN().to(device)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

def locally_connected(train_data: ZipDataset, test_data: ZipDataset) -> None:
    model = LocallyConnectedNN().to(device)
    train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=150).plot()

def ensemble(train_data: ZipDataset, test_data: ZipDataset) -> None:
    model1 = FullyConnectedNN().to(device)
    model2 = LocallyConnectedCNN().to(device)
    model3 = LocallyConnectedNN().to(device)
    metrics1, metrics2, metrics3, metrics4=trainEnsemble(model1, model2, model3, train_data, test_data, batch_size=128, learning_rate=0.005, epochsArray=[500,150,150])
    metrics1.plot()
    metrics2.plot()
    metrics3.plot()
    metrics4.plot()
if __name__ == "__main__":
    device = get_gpu()
    print(f"Using device: {device}")
    train_data = ZipDataset("../zip_train.txt")
    test_data = ZipDataset("../zip_test.txt")
    # fully_connected(train_data, test_data)
    # convolutional(train_data, test_data)
    # locally_connected(train_data, test_data)
    ensemble(train_data, test_data)
    plt.show()