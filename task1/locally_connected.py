import torch
import torch.nn as nn
from train import train
from zip_dataset import ZipDataset

class LocallyConnected2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LocallyConnected2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Create a set of weights for the local neighborhood; no shared weights
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # Get input dimensions (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = x.size()

        # Calculate the output height and width
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        # Initialize the output
        output = torch.zeros(batch_size, self.out_channels, output_height, output_width).to(x.device)

        # Perform the locally connected operation: sliding the kernel over the input
        for i in range(output_height):
            for j in range(output_width):
                # Extract the local patch for each sliding window
                local_patch = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]

                # Apply the weights (no shared weights) for each kernel in the local neighborhood
                for k in range(self.out_channels):
                    output[:, k, i, j] = (local_patch * self.weights[k]).sum(dim=(1, 2, 3))

        return output

class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()

        self.layer1 = LocallyConnected2D(1, 32, 3)
        self.layer2 = LocallyConnected2D(32, 64, 2)
        self.layer3 = LocallyConnected2D(64, 128, 1)
        # Fully connected layer for final output
        self.layer4 = nn.Linear(128 * 13 * 13, 10)  # Flattening the output to 10 classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Reshape the input to (batch_size, 1, 16, 16)
        x = x.view(x.size(0), 1, 16, 16)

        # Apply locally connected layers
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))

        # Flatten the output from the last layer and pass through the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.layer4(x)

        return x


if __name__ == "__main__":
    model = LocallyConnectedNN()
    train_data = ZipDataset("../zip_train.txt")
    test_data = ZipDataset("../zip_test.txt")

    metrics = train(model, train_data, test_data, 32, 0.01, 10)
    metrics.plot()

