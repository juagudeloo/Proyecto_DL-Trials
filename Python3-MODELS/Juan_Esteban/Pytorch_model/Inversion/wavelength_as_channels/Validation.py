import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

from muram import MuRAM
from inv_model_py import InvModel1

def main():
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ptm = "/media/juanessao2000/hdd/PRINCIPAL-2023-2/PROYECTOS/PROYECTO_DL/Numpy_MuRAM_data/"
    pth_out = "/kaggle/input/muram-data/"
    filename = "130000"
    
    #Create model save path 
MODEL_PATH = Path(pth_out)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "inversion1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


if __name__ == "__main__":
    main()