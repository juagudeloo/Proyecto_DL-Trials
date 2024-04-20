import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

from muram import MuRAM
from nn_model import InvModel1

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

    muram = MuRAM(ptm=ptm, pth_out=pth_out, filename=filename)
    atm_quant, stokes = muram.charge_quantities()
    generated_atm = np.zeros_like(atm_quant)

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")

    atm_quant = torch.from_numpy(atm_quant).to(device)
    atm_quant = torch.moveaxis(atm_quant, (1,2))
    atm_s = atm_quant.size()
    atm_quant = torch.reshape(atm_quant,(atm_s[0]*atm_s[1], atm_s[2], atm_s[3]))
    stokes = torch.from_numpy(stokes).to(device)

    print(atm_quant.size(), stokes.size())
    
    
    for i in range(muram.nx):
        for j in range(muram.nz):


if __name__ == "__main__":
    main()