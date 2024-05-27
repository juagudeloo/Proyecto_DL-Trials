import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

from muram import MuRAM
from nn_model import *
import os

def main():
    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    pth_out = "Results/"
    filename = "130000"

    muram = MuRAM(ptm = ptm, pth_out = pth_out, filename = filename)
    atm_quant, stokes = muram.charge_quantities()
    atm_quant = np.moveaxis(atm_quant, 1,2)
    stokes = np.reshape(stokes, (480*480,300,4))
    stokes = np.moveaxis(stokes, 1,2)

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #Create model save path 
    MODEL_PATH = Path(pth_out+"model_weights")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    nn_params = "10E0.0001lr"
    MODEL_NAME = "inversion_wave_chan_"+nn_params+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    loaded_model1 = InvModel1(4,4*20,4096).float().to(device)
    loaded_model1.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=device))

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")

    atm_quant = torch.from_numpy(atm_quant).to(device)
    stokes = torch.from_numpy(stokes).to(device)
    stokes_s = stokes.size()
    # stokes = torch.reshape(stokes,(stokes_s[0]*stokes_s[1], stokes_s[2], stokes_s[3]))
    with torch.inference_mode():
        loaded_model1.eval()
        generated_atm = loaded_model1.float()(stokes.float())
        generated_atm = generated_atm.to("cpu").numpy()
        generated_atm = np.reshape(generated_atm, (muram.nx, muram.nz, 20, 4))

    generated_out = pth_out+"Validation/"
    if not os.path.exists(generated_out):
        os.mkdir(generated_out)
    np.save(generated_out+"generated_atm"+f"_{filename}_"+nn_params, generated_atm)



if __name__ == "__main__":
    main()