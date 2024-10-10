import sys
import os

import time
import datetime

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

#Own modules
sys.path.append("/girg/juagudeloo/Proyecto_DL-Trials/Inversion_model_1D/module")
from utils.test_utils  import *
from utils.train_utils import create_model_save_path

images_out = "Results/Images/"

def plot_metrics(trl_path: str, 
                 tsl_path: str, 
                 tsacc_path: str) -> None:
    """
    Function to plot the metrics of the training.
    ---------------------------------------------

    Args:
        trl_path (str): Path to the training loss numpy file.
        tsl_path (str): Path to the test loss numpy file.
        tsacc_path (str): Path to the test accuracy numpy file.
    """
    
    loss_metrics = {}
    loss_metrics[0] = np.load(trl_path)
    loss_metrics[1] = np.load(tsl_path)
    loss_metrics[2] = np.load(tsacc_path)
    
    metrics_titles = ["Train loss", "Test loss", "Test accuracy [%]"]

    N_metrics = len(metrics_titles)
    N_epochs = len(loss_metrics[0])
    epochs_line = np.arange(1,N_epochs+1,1)
    
    fig, ax = plt.subplots(1,3,figsize=(3*4.5,1*3), layout = "constrained")

    for i in range(N_metrics):
        ax[i].plot(epochs_line, loss_metrics[i])
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metrics_titles[i])
        ax[i].set_xlim((0,N_epochs))
        if not i==2:
            ax[i].set_yscale("log")
    ax[2].set_ylim((80,100))
    
    metrics_img_out = images_out+"metrics/"
    fig.savefig(metrics_img_out)
    
def test_model(ptm:str, 
               model: nn.Module,
               lr: float,
               epochs: int,
               filename: str,
               batch_size: int,
               vertical_comp: bool
               ) -> None:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    MODEL_SAVE_PATH = create_model_save_path(epochs, lr)
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=device))
    
    
    # Load the data
    stokes, original_atm, generated_atm = generate_new_data(ptm,
                                                    model,
                                                    filename,
                                                    batch_size,
                                                    vertical_comp)
    # Save pixel plots
    pixels = [[90,250],
              [100,115]]
    for pix in pixels:
        plot_pixel(filename,
                   stokes, 
                   original_atm,
                   generated_atm, 
                   *pix)
    
    # Save correlation plots
    heights = [1, 5, 10, 14]
    for iheight in heights:
        plot_corr_diff_OD(filename,
                          original_atm, 
                          generated_atm, 
                          iheight)
    
    # Save depth dependent error plots
    all_depth_error(filename,
                    original_atm,
                    generated_atm)
    
    
    
            
        