#This new main saves all the files data in the same TestDataset.
import torch
from torch import nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

sys.path.append("../../module")
from muram import MuRAM
from training import train_model
from testing import plot_metrics
from nn_model import *


def main():
    #Creating the model for training
    ptm = "/scratchsan/observatorio/juagudeloo/data/"
    model = InvModel1(36,6*20,hidden_units=4096).float()
    batch_size = 80
    vertical_comp = False
    
    lr = 1e-5
    epochs = 20
    
    plot_metrics(
        *train_model(ptm, 
                     model, 
                     lr,
                     epochs,
                     batch_size,
                     vertical_comp,
                     opt_depth_stratif=True
                     )
        )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    

#valid
if __name__ == "__main__":
    main()
