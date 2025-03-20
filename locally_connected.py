import torch
import torch.nn as nn
from zip_dataset import ZipDataset
from train import train, evaluate, get_gpu 

class LocallyConnectedLayer(nn.Module):
    """
    A locally connected layer where each spatial region has its own independent weights (no shared weights).
    Optimized for GPU usage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, input_size):
        super(LocallyConnectedLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size

        # Compute output spatial dimensions
        self.output_height = input_size[0] - kernel_size + 1
        self.output_width = input_size[1] - kernel_size + 1

        # Learnable weights & biases for each position (No Shared Weights)
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, self.output_height, self.output_width, kernel_size, kernel_size)
        )
        self.biases = nn.Parameter(torch.zeros(out_channels, self.output_height, self.output_width))

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()

        x_patches = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        # Expand dimensions for broadcasting
        x_patches = x_patches.unsqueeze(1)
        weights = self.weights.unsqueeze(0)

        # Element-wise multiplication and summation over patch dimensions
        output = (x_patches * weights).sum(dim=(2, 5, 6)) + self.biases.unsqueeze(0)

        return output


class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()

        # First three layers: Locally Connected (No Shared Weights)
        self.local1 = LocallyConnectedLayer(1, 16, 3, (16, 16))
        self.local2 = LocallyConnectedLayer(16, 32, 3, (14, 14))
        self.local3 = LocallyConnectedLayer(32, 64, 3, (12, 12))

        # Batch Normalization
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)

        # Activation Functions
        self.relu = nn.ReLU()  # Required in at least one layer
        self.tanh = nn.Tanh()  # Required in at least one layer

        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 10 * 10, 10)  # 10 output classes

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.relu(self.batchnorm1(self.local1(x)))
        x = self.relu(self.batchnorm2(self.local2(x)))
        x = self.tanh(self.batchnorm3(self.local3(x)))

        x = x.view(x.size(0), -1)  # Flatten before FC layer
        x = self.fc1(x)
        return x



if __name__ == "__main__":
    device = get_gpu()
    print(f"Using device: {device}")

    train_data = ZipDataset("zip_train.txt")
    test_data = ZipDataset("zip_test.txt")

    print("\nTraining Locally Connected NN...")
    model_local_nn = LocallyConnectedNN().to(device)  

    metrics = train(model_local_nn, train_data, test_data, batch_size=512, learning_rate=0.005, epochs=10)
    metrics.plot()

    print("\n=== Final Test Evaluation ===")
    avg_loss, accuracy, error_rate = evaluate(model_local_nn, test_data)
    print(f"Locally Connected NN -> Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

