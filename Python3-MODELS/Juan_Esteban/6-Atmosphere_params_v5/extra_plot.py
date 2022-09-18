import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from data_class import Data_class, inverse_scaling

def main():
    obtained_file = int(input("File of the obtained atmosphere values: "))
    filename = "0"+str(obtained_file)
    atm_params = np.load(f"/mnt/scratch/juagudeloo/obtained_data/obtained_value-0{obtained_file}.npy")
    data = Data_class(create_scaler=False)
    original_atm = data.charge_atm_params(filename)
    scaler_names = ["mbyy", "mvyy", "mrho", "mtpr"]
    for i in range(len(scaler_names)):
        original_atm[:,:,i,:] = np.memmap.reshape(inverse_scaling(original_atm[:,:,i,:], scaler_names[i]), (480,480,(256-180)))


    height = 10

    titles = ["Magnetic field LOS", "Velocity LOS", "Density", "Temperature"]
    ylabels = [r"$B_z$ [G]", r"$v$ [$10^5$ cm s$^{-1}$]", r"$T$ [K]", r"$\rho$[g cm$^{-3}$]"]
    fig, ax = plt.subplots(3,4,figsize=(7,7))

    for i in range(4):
        im_i = ax[i].imshow(original_atm[:,:,i,height], cmap="gist_gray")
        ax[i].set_title(ylabels[i])
        ax[i].legend()
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('bottom', size='5%', pad=0.3)
        fig.colorbar(im_i, cax=cax, orientation="horizontal")
    fig.savefig(f"magnitude_B_{filename}.png")
if __name__ == "__main__":
    main()