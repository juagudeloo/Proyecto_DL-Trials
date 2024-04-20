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

    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    pth_out = "Results/"
    filename = "130000"
    
    #Create model save path 
    MODEL_PATH = Path(pth_out+"model_weights")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "inversion1.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    loaded_model1 = InvModel1(300,4*20,4096).float().to(device)
    loaded_model1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    muram = MuRAM(ptm=ptm, pth_out=pth_out)
    muram.charge_quantities()


if __name__ == "__main__":
    main()