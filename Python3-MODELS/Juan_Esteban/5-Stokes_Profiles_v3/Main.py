import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_NN_model

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    filename = "053000"
    #Stokes parameters specifications
    stokes_ptm = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    stokes_filename = filename+"_0000_0000.prof"
    model = Data_NN_model()
    iout_ravel, iout = model.charge_intensity(ptm, filename)
    profs = model.charge_stokes_params(stokes_ptm, stokes_filename)
    print(np.shape(iout))
    print(np.shape(profs))


    title = ['I','Q','U','V']

if __name__ == "__main__":
    main()