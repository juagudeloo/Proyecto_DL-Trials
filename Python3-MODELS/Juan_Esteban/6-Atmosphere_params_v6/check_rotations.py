from train_generate.extract_compile.data_class import Data_class
import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = "099000"
    sun = Data_class()
    atm_params = sun.charge_atm_params(filename)
    atm_titles = ["mbyy", "mvyy", "mrho", "mtpr"]
    intensity = sun.charge_intensity(filename)
    stokes = sun.charge_stokes_params(filename)
    stokes_titles = ["I", "Q", "U", "V"]
    print(np.shape(atm_params))
    print(np.shape(intensity))
    print(np.shape(stokes))

    fig, ax = plt.subplots(3,4, figsize = (32,7))
    for i in range(4):
        ax[0,i].imshow(atm_params[:,:,10,i])
        ax[0,i].set_title(atm_titles[i])
        
        ax[2,i].imshow(stokes[:,:,10,i])
        ax[2,i].set_title(stokes_titles[i])

    ax[1,0].imshow(intensity)
    ax[1,0].set_title("Intensity")

    fig.savefig("Images/rotations.png")


if __name__ == "__main__":
    main()
