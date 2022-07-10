import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_NN_model

def main():
    n_file = "053000"
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    filename = n_file
    #Stokes parameters specifications
    stokes_ptm = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    stokes_filename = n_file+"_0000_0000.prof"
    model = Data_NN_model()
    profs = model.charge_stokes_params(stokes_ptm, stokes_filename)
    print(np.shape(profs))

    title = ['I','Q','U','V']
    fig, ax = plt.subplots(1,4,figsize=(40,10))
    for i in range(4):
        ax[i].imshow(profs[:,:,100,i], "gist_gray")
        ax[i].set_title(title[i])
    fig.savefig("Images/stk_spatial_wl=ctn.png")

    fig, ax = plt.subplots(1,4,figsize=(40,10))
    for i in range(4):
        ax[i].plot(range(len(profs[0,0,:,i])), profs[0,0,:,i])
        ax[i].set_title(title[i])
    fig.savefig("Images/stk_vs_wl.png")

if __name__ == "__main__":
    main()