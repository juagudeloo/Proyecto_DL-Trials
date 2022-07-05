import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_NN_model

def main():
    #Stokes parameters specifications
    stokes_ptm = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    stokes_filename = "053000_0000_0000.prof"
    model = Data_NN_model()
    profs = model.charge_stokes_params(stokes_ptm, stokes_filename)
    print(np.shape(profs))


if __name__ == "__main__":
    main()