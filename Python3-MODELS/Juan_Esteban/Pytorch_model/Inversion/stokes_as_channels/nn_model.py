from torch import nn
import torch

class InvModel1(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_units):
        super().__init__()
        padding = 1
        self.simple_conv = nn.Sequential(nn.Conv1d(in_channels=in_shape, out_channels=512, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=512, out_channels=512, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=512, out_channels=512, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=512, out_channels=512, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=512, out_channels=512, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Flatten(),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features = 77312, out_features = out_shape))
    def forward(self, x):
        return self.simple_conv(x)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time