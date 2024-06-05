from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class InvModel1(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_units):
        super().__init__()
        padding = 1
        self.simple_conv = nn.Sequential(nn.Conv1d(in_channels=in_shape, out_channels=600, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=600, out_channels=1200, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=1200, out_channels=2400, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=2400, out_channels=4800, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features = 38400, out_features = out_shape))
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

def validation_visual(ref_quant_list:list, generated_quant:np.ndarray, epochs_to_plot:list, images_out:str, titles:list):
    """
    Function for making the correlation plots.
    ------------------------------------------------
    ref_quant_list (list): list of the generated params for the specified epochs.
    generated_quant (np.ndarray): reference cube atmosphere params.
    epochs_to_plot (list): list with the name of the filename along with the epoch of training.
    images_out (str): path to save the animation.
    title (list): list of the titles corresponding to the plotted magnitudes.
    """
    N_plots = generated_quant.shape[-1]
    heights_index = [11, 8, 5, 2]
    N_heights = len(heights_index)
    def animate(ni):
        tau = np.linspace(-3, 1,20)
        ref_quant = ref_quant_list[ni]
        for i in range(4):
            for j in range(N_plots):
                ax[i,j].scatter(generated_quant[:,heights_index[j],:,i].flatten(), 
                                ref_quant[:,:,heights_index[j],i].flatten(),
                                s=5, c="darkviolet", alpha=0.1)
                ax[i,j].set_title(f"{titles[j]} - OD={tau[heights_index[i]]} - E={epochs_to_plot[ni]}")
                ax[i,j].set_xlabel("generated")
                ax[i,j].set_ylabel("reference")
            
    fig, ax = plt.subplots(N_heights, N_plots, figsize=(4*N_plots, 4*N_heights))
    fig.tight_layout()
    frames = len(epochs_to_plot)
    animator = animation.FuncAnimation(fig, animate, frames=frames)
    animator.save(images_out+"visualization.mp4")
