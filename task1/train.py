import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from zip_dataset import ZipDataset

class TrainingMetrics:
    """Stores metrics per epoch during training. Can then plot the metrics."""

    def __init__(self) -> None:
        self.epochs = 0

        # Team: If you can think of any other metrics, feel free to add.
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
        plt.show()


def train(model: nn.Module, train_data: ZipDataset, test_data: ZipDataset, batch_size: int, learning_rate: float, \
          epochs: int = 500, momentum: float = 0) -> TrainingMetrics:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    metrics = TrainingMetrics()

    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        metrics.add_epoch(total_loss / len(train_data))

        if epoch % (epochs // 10) == 0:
            print(f"Training {model.__class__.__name__} model: {epoch}/{epochs} epochs")

            avg_loss, accuracy, error_rate = evaluate(model, test_data)
            print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

    print(f"Training {model.__class__.__name__} finished.")
    avg_loss, accuracy, error_rate = evaluate(model, test_data)
    print(f"Avg. Loss: {avg_loss:.5}, Accuracy: {accuracy:.5}%, Error Rate: {error_rate:.5}%")

    return metrics


def evaluate(model: nn.Module, test_data: ZipDataset):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    num_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, label in test_data:
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss.item()

            predicted_label = torch.argmax(output)
            num_correct += (label == predicted_label).item()

    avg_loss = total_loss / len(test_data)
    accuracy = 100 * num_correct / len(test_data)
    error_rate = 100 - accuracy

    model.train()

    return avg_loss, accuracy, error_rate

