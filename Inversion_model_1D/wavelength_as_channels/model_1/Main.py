#This new main saves all the files data in the same TestDataset.
import torch
from torch import nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

sys.path.append("/girg/juagudeloo/Proyecto_DL-Trials/Inversion_model_1D/module")
from muram import MuRAM
from training import train_model
from nn_model import *


def main():
    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    pth_out = "Results/"
    training_files = ["085000", 
    "090000","095000", "100000", "105000", "110000"
    ]

    #Creating the model for training
    model = InvModel1(36,6*20,4096).float()

    #Model training hyperparams
    vertical_comp = False # Whether to use or not just the LOS components.
    loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places
    lr = 5e-5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    epochs = 12
    batch_size = 80
    
    train_model(model = model,
                training_files = training_files,
                ptm = ptm,
                pth_out = pth_out,
                epochs = epochs, 
                lr = lr, 
                batch_size = batch_size,
                loss_fn = loss_fn, 
                optimizer = optimizer,
                vertical_comp = vertical_comp)
    
    
    
    
    
    


    
    

#valid
if __name__ == "__main__":
    main()
