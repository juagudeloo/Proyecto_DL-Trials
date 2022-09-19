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
    max_height = len(atm_params[0,0,0,:])
    max_values = {}
    min_values = {}
    for i in range(4):
        max_values[i] = np.argwhere(atm_params[:,:,i,height]==np.max(atm_params[:,:,i,height]))
        min_values[i] = np.argwhere(atm_params[:,:,i,height]==np.min(atm_params[:,:,i,height]))
    
    #Location values obtained from the temperature data
    max_x_plot = max_values[3][0][0]
    max_z_plot = max_values[3][0][1]
    min_x_plot = min_values[3][0][0]
    min_z_plot = min_values[3][0][1]

    titles = ["Magnetic field LOS", "Velocity LOS", "Density", "Temperature"]
    ylabels = [r"$B_z$ [G]", r"$v$ [$10^5$ cm s$^{-1}$]", r"$T$ [K]", r"$\rho$[g cm$^{-3}$]"]
    fontsize = 16
    
    fig, ax = plt.subplots(2,2,figsize=(15,15))
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(0,max_height,1)+1, atm_params[max_x_plot, max_z_plot, i+j], label = "generated params")
            ax[i,j].plot(np.arange(0,max_height,1)+1, original_atm[max_x_plot, max_z_plot, i+j], label = "original params")
            ax[i,j].set_title("In maximum", fontsize = fontsize)
            ax[i,j].legend(fontsize = fontsize)
            ax[i,j].set_xlabel("height pixels", fontsize = fontsize)
            ax[i,j].set_ylabel(ylabels[i+j], fontsize = fontsize)
            ax[i,j].ticklabel_format(style = "sci")
    fig.savefig(f"Images/Stokes_params/height_serie_plots_0{obtained_file}-00.png")

    fig, ax = plt.subplots(2,2,figsize=(15,15))
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(0,max_height,1)+1, atm_params[min_x_plot, min_z_plot, i+j], label = "generated params")
            ax[i,j].plot(np.arange(0,max_height,1)+1, original_atm[min_x_plot, min_z_plot, i+j], label = "original params")
            ax[i,j].set_title("In minimum", fontsize = fontsize)
            ax[i,j].legend(fontsize = fontsize)
            ax[i,j].set_xlabel("height pixels", fontsize = fontsize)
            ax[i,j].set_ylabel(ylabels[i+j], fontsize = fontsize)
            ax[i,j].ticklabel_format(style = "sci")
    fig.savefig(f"Images/Stokes_params/height_serie_plots_0{obtained_file}-01.png")

    fig, ax = plt.subplots(2,2,figsize=(11,11))
    for i in range(2):
        for j in range(2):
            im_i = ax[i,j].imshow(atm_params[:,:,i+j,height], cmap="gist_gray")
            ax[i,j].scatter(max_x_plot, max_z_plot, label = "maximum", color = "r")
            ax[i,j].scatter(min_x_plot, min_z_plot, label = "minimun", color = "g")
            ax[i,j].set_title(ylabels[i+j], fontsize = fontsize)
            ax[i,j].legend(fontsize = fontsize)
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes('bottom', size='5%', pad=0.3)
            fig.colorbar(im_i, cax=cax, orientation="horizontal")
    fig.savefig(f"Images/Stokes_params/height_serie_plots_0{obtained_file}-02.png")
    
#


if __name__ == "__main__":
    main()