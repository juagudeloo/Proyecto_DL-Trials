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
from testing import test_model
from utils.train_utils  import create_model_save_path
from nn_model import *


def main():
    #Creating the model for training
    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    #Creating the model for training
    model = InvModel1(36,6*20,4096).float()
    lr = 5e-5
    epochs = 12
    
    MODEL_SAVE_PATH = create_model_save_path(epochs, lr)
    
    filename = "175000"
    batch_size = 80
    vertical_comp = False
    
    test_model(ptm, 
               model, 
               filename,
               batch_size,
               vertical_comp)
    