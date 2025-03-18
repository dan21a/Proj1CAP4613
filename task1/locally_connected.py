import torch
import torch.nn as nn
import torch.optim as optim
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
        output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device)

        # Vectorized locally connected operation
        for i in range(output_height):
            for j in range(output_width):
                local_patch = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                output[:, :, i, j] = (local_patch.unsqueeze(1) * self.weights).sum(dim=(2, 3, 4))

        return output

class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()

        self.layer1 = LocallyConnected2D(1, 16, 3)  # Reduce filters to speed up
        self.layer2 = LocallyConnected2D(16, 32, 2)
        self.layer3 = LocallyConnected2D(32, 64, 1)
        self.layer4 = nn.Linear(64 * 13 * 13, 10)
        
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Helps prevent overfitting
    
    def forward(self, x):
        x = x.view(x.size(0), 1, 16, 16)
        x = self.relu(self.batchnorm1(self.layer1(x)))
        x = self.relu(self.batchnorm2(self.layer2(x)))
        x = self.relu(self.batchnorm3(self.layer3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.layer4(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LocallyConnectedNN().to(device)
    train_data = ZipDataset(r"C:\Users\niebl\Downloads\zip_train.txt")
    test_data = ZipDataset(r"C:\Users\niebl\Downloads\zip_test.txt")

    # Train model
    metrics = train(model, train_data, test_data, batch_size=128, learning_rate=0.005, epochs=5)
    metrics.plot()

    # Explicitly run testing after training finishes
    from train import evaluate  # Ensure you import this function

    print("\n=== Running Final Test Evaluation ===")
    avg_loss, accuracy, error_rate = evaluate(model, test_data)
    print(f"Final Test Results -> Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

