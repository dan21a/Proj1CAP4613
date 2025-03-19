import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from zip_dataset import ZipDataset
from train import train, evaluate, get_gpu


# ------------------ Locally Connected Convolutional Network (CNN with Shared Weights) -------------------
class LocallyConnectedCNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedCNN, self).__init__()

        # Convolutional layers (Locally connected with shared weights)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 32 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 filters

        # Batch Normalization
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)

        # Activation Functions
        self.relu = nn.ReLU()  # Required in at least one layer
        self.tanh = nn.Tanh()  # Required in at least one layer

        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 16 * 16, 10)  # 10 output classes

    def forward(self, x):
        x = x.view(-1, 1, 16, 16)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.tanh(self.batchnorm3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten before FC layer
        x = self.fc1(x)
        return x


# ------------------ Run Training on GPU -------------------
if __name__ == "__main__":
    device = get_gpu()
    print(f"Using device: {device}")

    train_data = ZipDataset("../zip_train.txt")
    test_data = ZipDataset("../zip_test.txt")

    print("\nTraining Locally Connected CNN With Shared Weights on GPU...")
    model_local_cnn = LocallyConnectedCNN().to(device)

    metrics = train(model_local_cnn, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=200)
    metrics.plot() 

    print("\n=== Final Test Evaluation ===")
    avg_loss, accuracy, error_rate = evaluate(model_local_cnn, test_data)
    print(f"Locally Connected CNN -> Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

