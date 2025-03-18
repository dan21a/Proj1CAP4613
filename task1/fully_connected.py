import torch.nn as nn
from torch import Tensor
from task1.train import train
from zip_dataset import ZipDataset

class FullyConnectedNN(nn.Module):
    def __init__(self) -> None:
        super(FullyConnectedNN, self).__init__()

        # We have four layers, as required.
        # First layer has 256 neurons for each pixel of an image.
        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 10)

        # We use relu and sigmoid, as required.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    model = FullyConnectedNN()
    train_data = ZipDataset(r"C:\Users\niebl\Downloads\zip_train.txt")
    test_data = ZipDataset(r"C:\Users\niebl\Downloads\zip_test.txt")

    metrics = train(model, train_data, test_data, 32, 0.01)
    metrics.plot()

