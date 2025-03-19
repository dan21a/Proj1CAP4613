import matplotlib.pyplot as plt
import torch
from torch.mps import is_available
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from zip_dataset import ZipDataset
import copy

def get_gpu() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class TrainingMetrics:
    """Stores metrics per epoch during training. Can then plot the metrics."""

    def __init__(self) -> None:
        self.epochs = 0
        self.loss = []

    def add_epoch(self, loss: float) -> None:
        self.epochs += 1
        self.loss.append(loss)

    def plot(self) -> None:
        plt.figure()
        plt.plot(range(self.epochs), self.loss, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss During Training")
        plt.show(block=False)


def train(model: nn.Module, train_data: ZipDataset, test_data: ZipDataset, batch_size: int, learning_rate: float, \
        epochs: int = 500, momentum: float = 0) -> TrainingMetrics:
    device = get_gpu()
    model.to(device)  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    metrics = TrainingMetrics()

    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device) 
            optimizer.zero_grad()
            output = model(data.view(-1, 1, 16, 16))  
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        metrics.add_epoch(total_loss / len(train_data))

        if epoch % max(epochs // 10, 1) == 0:
            print(f"Training {model.__class__.__name__} model: {epoch}/{epochs} epochs")

            avg_loss, accuracy, error_rate = evaluate(model, test_data)
            print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

    print(f"Training {model.__class__.__name__} finished.")
    avg_loss, accuracy, error_rate = evaluate(model, test_data)
    print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

    return metrics

def trainEnsemble(model1: nn.Module, model2: nn.Module, model3: nn.Module, train_data: ZipDataset, test_data: ZipDataset, batch_size: int, learning_rate: float, \
        epochs: int = 500, momentum: float = 0) -> TrainingMetrics:
    device = get_gpu()
    model1.to(device)  
    model2.to(device)  
    model3.to(device)  

    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=momentum)
    optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=momentum)
    optimizer3 = optim.SGD(model3.parameters(), lr=learning_rate, momentum=momentum)

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    metrics1 = TrainingMetrics()
    metrics2 = TrainingMetrics()
    metrics3 = TrainingMetrics()

    for epoch in range(epochs):
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device) 
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            output1 = (model1(data.view(-1, 1, 16, 16))+model2(data.view(-1, 1, 16, 16))+model3(data.view(-1, 1, 16, 16)))/3
            output2 = copy.copy(output1)
            output3 = copy.copy(output1)
            loss1 = criterion(output1, labels)
            loss2 = criterion(output2, labels)
            loss3 = criterion(output3, labels)
            loss1.backward()
            loss2.backward()
            loss3.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            total_loss1 += criterion(model1(data.view(-1, 1, 16, 16)), labels).item() * batch_size
            total_loss2 += criterion(model2(data.view(-1, 1, 16, 16)), labels).item() * batch_size
            total_loss3 += criterion(model3(data.view(-1, 1, 16, 16)), labels).item() * batch_size

        metrics1.add_epoch(total_loss1 / len(train_data))
        metrics2.add_epoch(total_loss2 / len(train_data))
        metrics3.add_epoch(total_loss3 / len(train_data))

        if epoch % max(epochs // 10, 1) == 0:
            print(f"Training: {epoch}/{epochs} epochs")
            avg_loss, accuracy, error_rate = evaluate(model1, test_data)
            print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")
            avg_loss, accuracy, error_rate = evaluate(model2, test_data)
            print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")
            avg_loss, accuracy, error_rate = evaluate(model3, test_data)
            print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

    print(f"Training finished.")
    avg_loss, accuracy, error_rate = evaluate(model1, test_data)
    print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")
    avg_loss, accuracy, error_rate = evaluate(model2, test_data)
    print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")
    avg_loss, accuracy, error_rate = evaluate(model3, test_data)
    print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")
    return metrics1, metrics2, metrics3

def evaluate(model: nn.Module, test_data: ZipDataset):
    device = get_gpu()
    model.eval()

    criterion = nn.CrossEntropyLoss()
    num_correct = 0
    total_loss = 0.0

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)  
            output = model(data.view(-1, 1, 16, 16))  
            loss = criterion(output, label)
            total_loss += loss.item()

            predicted_label = torch.argmax(output)
            num_correct += (label == predicted_label).item()

    avg_loss = total_loss / len(test_data)
    accuracy = 100 * num_correct / len(test_data)
    error_rate = 100 - accuracy

    model.train()
    return avg_loss, accuracy, error_rate
