import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from zip_dataset import ZipDataset
from train import train, evaluate 

# ------------------ Locally Connected Network (No Shared Weights in First 3 Layers) -------------------
class LocallyConnectedLayer(nn.Module):
    """
    A locally connected layer where each spatial region has its own independent weights (no shared weights).
    """
    def __init__(self, in_channels, out_channels, kernel_size, input_size):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the local receptive field
            input_size: Tuple (height, width) of the input image
        """
        super(LocallyConnectedLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size

        # Calculate the output height and width
        self.output_height = input_size[0] - kernel_size + 1
        self.output_width = input_size[1] - kernel_size + 1

        # Define learnable weights for each spatial position (no sharing)
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, self.output_height, self.output_width, kernel_size, kernel_size))
        self.biases = nn.Parameter(torch.zeros(out_channels, self.output_height, self.output_width))

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        output = torch.zeros(batch_size, self.out_channels, self.output_height, self.output_width, device=x.device)

        # Manually apply the local connection (no shared weights)
        for i in range(self.output_height):
            for j in range(self.output_width):
                local_patch = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]  # Extract local region
                output[:, :, i, j] = (local_patch.unsqueeze(1) * self.weights[:, :, i, j, :, :]).sum(dim=(2, 3, 4)) + self.biases[:, i, j]

        return output


class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()

        # First three layers: Locally Connected (No Shared Weights)
        self.local1 = LocallyConnectedLayer(1, 16, 3, (16, 16))  # 16 filters
        self.local2 = LocallyConnectedLayer(16, 32, 3, (14, 14))  # 32 filters
        self.local3 = LocallyConnectedLayer(32, 64, 3, (12, 12))  # 64 filters

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
        x = x.view(-1, 1, 16, 16)  # ✅ Fix input shape
        x = self.relu(self.batchnorm1(self.local1(x)))
        x = self.relu(self.batchnorm2(self.local2(x)))
        x = self.tanh(self.batchnorm3(self.local3(x)))

        x = x.view(x.size(0), -1)  # Flatten before FC layer
        x = self.fc1(x)
        return x


# ------------------ Run Training on GPU -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = ZipDataset(r"C:\Users\niebl\Downloads\zip_train.txt")
    test_data = ZipDataset(r"C:\Users\niebl\Downloads\zip_test.txt")

    print("\nTraining Locally Connected NN on GPU...")
    model_local_nn = LocallyConnectedNN().to(device)

    metrics = train(model_local_nn, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=10)

    print("\n=== Final Test Evaluation ===")
    avg_loss, accuracy, error_rate = evaluate(model_local_nn, test_data)
    print(f"Locally Connected NN -> Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")
