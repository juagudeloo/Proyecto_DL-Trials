import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_NN_model

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    filename = ["053000", "056000"]
    #Stokes parameters specifications
    stokes_ptm = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    stokes_filename = []
    for elem in filename:
        stokes_filename.append(elem+"_0000_0000.prof")
    model = Data_NN_model()
    model.charge_inputs(ptm, filename)
    iout = model.charge_intensity(ptm, filename)
    profs = model.charge_stokes_params(stokes_ptm, stokes_filename)
    print(np.shape(iout))
    print(np.shape(profs))


    fig, ax = plt.subplots(1,2,figsize=(20,10))
    title = ['I','Q','U','V']
    for i in range(len(filename)):
        ax[i,0].imshow(iout[i])
        ax[i,1].imshow(profs[i][:,:,0,0])
        fig.savefig("Images/iout_vs_stokes.png")

if __name__ == "__main__":
    main()