import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Custom dataset loader for ZIP dataset
class ZIPDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.labels = []
        with open(file_path, 'r') as f:
            for line in f:
                values = list(map(float, line.split()))
                self.labels.append(int(values[0]))
                self.data.append(np.array(values[1:]))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Load dataset
def get_data_loaders(batch_size=32):
    train_dataset = ZIPDataset(r"C:\Users\niebl\Downloads\zip_train.txt")
    test_dataset = ZIPDataset(r"C:\Users\niebl\Downloads\zip_test.txt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

# Define the Fully Connected Network (FCN)
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Locally Connected Network (LCN)
class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(-1, 1, 16, 16)  # Reshape input to (batch_size, 1, 16, 16)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Hyperparameter Initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Training function
def train_model(model, train_loader, learning_rate=0.01, momentum=0.9, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print(f"Training complete for {model.__class__.__name__}")

# Function to print class distributions
def print_class_distribution(dataset, dataset_name):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(dataset)

    # Convert to standard Python types
    counts_dict = {int(k): int(v) for k, v in zip(unique, counts)}
    proportions_dict = {int(k): round(float(v), 2) for k, v in zip(unique, proportions)}

    print(f"Class distribution for {dataset_name}:")
    print("Counts:", counts_dict)
    print("Proportions:", proportions_dict)
    print()


# Running experiments
if __name__ == "__main__":
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders()
    
    
    
    models = [FullyConnectedNN(), LocallyConnectedNN(), CNN()]
    
    for model in models:
        model.apply(initialize_weights)
        train_model(model, train_loader)

    print_class_distribution(train_dataset, "Train")
    print_class_distribution(test_dataset, "Test")