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
    #Creating the model for training
    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    model = InvModel1(36,6*20,4096).float()
    batch_size = 80
    vertical_comp = False
    
    lr = 5e-5
    epochs = 12
    
    #Creating the model for training
    model = InvModel1(36,6*20,4096).float()
    
    plot_metrics(
        *train_model(ptm, 
                     model, 
                     lr,
                     epochs,
                     batch_size,
                     vertical_comp)
        )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    

#valid
if __name__ == "__main__":
    main()
