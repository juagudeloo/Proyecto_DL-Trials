from train_generate.extract_compile.data_class import Data_class
import numpy as np

def main():
    filename = "092000"
    sun = Data_class()
    atm_params = sun.charge_atm_params(filename)
    intensity = sun.charge_intensity(filename)
    stokes = sun.charge_stokes_params(filename)
    print(np.shape(atm_params))
    print(np.shape(intensity))
    print(np.shape(stokes))

    fig, ax = plt.subplots(3,4, figsize = (32,7))
    ax[0,0].imshow(atm_params[:,:,10,0])
    ax[0,1].imshow(atm_params[:,:,10,1])
    ax[0,2].imshow(atm_params[:,:,10,2])
    ax[1,0].imshow(intensity)
    ax[2,0].imshow(stokes[:,:,10,0])
    ax[2,0].imshow(stokes[:,:,10,1])
    ax[2,0].imshow(stokes[:,:,10,2])
    ax[2,0].imshow(stokes[:,:,10,3])
|

if __name__ == "__main__":
    main()
