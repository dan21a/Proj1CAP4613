import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ZipDataset(Dataset):
    """
    Given a file_path to a normalized handwritten digit dataset in gzip
    format, creates a PyTorch DataSet from it. Each sample is a 256 long
    PyTorch float tensor representing the 16x16 image. 
    """

    def __init__(self, file_path: str) -> None:
        self.data = []
        self.labels = []

        with open(file_path, "r") as f:
            for line in f:
                # Parse the line as just a bunch of floats.
                values = list(map(float, line.split()))
                # The first value is the image's label.
                self.labels.append(torch.tensor(int(values[0])))
                # The remaining values are the image data.
                self.data.append(torch.tensor(values[1:], dtype=torch.float32))
                assert(len(self.data[-1]) == 256)

        print(f"Successfully created dataset from {file_path}")
        self.print_label_distribution()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx], self.labels[idx]

    def print_label_distribution(self) -> None:
        print("        " + " ".join(f"{n:>4}" for n in range(10)) + " Total")
        print("Counts: " + " ".join(f"{self.labels.count(n):>4}" for n in range(10)) + f" {len(self.labels):>4}")
        print("Ratios: " + " ".join(f"{self.labels.count(n) / len(self.labels):>4.1}" for n in range(10)))
        print()


def plot_zip(sample: Tensor) -> None:
    """
    Plots the given sample zip image for debugging.
    Sample should be a 256 long Tensor.
    """

    image_data = sample.numpy(force=True).reshape(16, 16)

    plt.imshow(image_data, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_zips(samples: list[Tensor]) -> None:
    """
    Plots the given sample zip images for debugging in a single plot.
    Samples should be 256 long Tensors.
    E.g., plot_zips(zip_dataset.data[1000:1500])
    """

    num_columns = int(np.ceil(np.sqrt(len(samples))))
    num_rows = int(np.ceil(len(samples) / num_columns))

    _, axes = plt.subplots(num_rows, num_columns)
    axes = np.array(axes).reshape(num_columns * num_rows)
    for i, ax in enumerate(axes):
        if i < len(samples):
            image_data = samples[i].numpy(force=True).reshape(16, 16)
            ax.imshow(image_data, cmap="gray")
        ax.axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

