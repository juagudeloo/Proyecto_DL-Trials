import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from data_class import Data_class


def main():
    obtained_file = int(input("File of the obtained atmosphere values: "))
    filename = "0"+str(obtained_file)
    stokes = np.load(f"/mnt/scratch/juagudeloo/obtained_data/Stokes_obtained_values-0{obtained_file}.npy")
    data = Data_class()
    original_stokes = data.charge_stokes_params(filename)


    wavelength = 10
    max_height = len(stokes[0,0,0,:])
    max_values = {}
    min_values = {}
    for i in range(4):
        max_values[i] = np.argwhere(stokes[:,:,i,wavelength]==np.max(stokes[:,:,i,wavelength]))
        min_values[i] = np.argwhere(stokes[:,:,i,wavelength]==np.min(stokes[:,:,i,wavelength]))
    
    #Location values obtained from the temperature data
    max_x_plot = max_values[3][0][0]
    max_z_plot = max_values[3][0][1]
    min_x_plot = min_values[3][0][0]
    min_z_plot = min_values[3][0][1]

    ylabels = [r"$I_{NORMALIZED}$", r"$Q_{NORMALIZED}$", r"$U_{NORMALIZED}$", r"$V_{NORMALIZED}$"]
    figsize = (13,13)
    fig, ax = plt.subplots(2,2,figsize=figsize)
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(6302,6302+10*300, 10), stokes[max_x_plot, max_z_plot, i*2+j], label = "generated stokes")
            ax[i,j].plot(np.arange(6302,6302+10*300, 10), original_stokes[max_x_plot, max_z_plot, i*2+j], label = "original stokes")
            ax[i,j].set_title("In maximum", fontsize = 16)
            ax[i,j].legend(fontsize = 16)
            ax[i,j].set_xlabel("height pixels", fontsize = 16)
            ax[i,j].set_ylabel(ylabels[i*2+j], fontsize = 16)
            ax[i,j].ticklabel_format(style = "sci")
            ax[i,j].tick_params(labelsize = 14)
    fig.savefig(f"Images/Stokes_params/stokes_plot_0{obtained_file}-01.png")

    fig, ax = plt.subplots(2,2,figsize=figsize)

    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(6302,6302+10*300, 10), stokes[max_x_plot, max_z_plot, i*2+j], label = "generated stokes")
            ax[i,j].set_title("In maximum", fontsize = 16)
            ax[i,j].legend(fontsize = 16)
            ax[i,j].set_xlabel("height pixels", fontsize = 16)
            ax[i,j].set_ylabel(ylabels[i*2+j], fontsize = 16)
            ax[i,j].ticklabel_format(style = "sci")
            ax[i,j].tick_params(labelsize = 14)
    fig.savefig(f"Images/Stokes_params/stokes_plot_0{obtained_file}-02.png")

    fig, ax = plt.subplots(2,2,figsize=figsize)


    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(6302,6302+10*300, 10), original_stokes[max_x_plot, max_z_plot, i*2+j], label = "original stokes", color = "orange")
            ax[i,j].set_title("In maximum", fontsize = 16)
            ax[i,j].legend(fontsize = 16)
            ax[i,j].set_xlabel("height pixels", fontsize = 16)
            ax[i,j].set_ylabel(ylabels[i*2+j], fontsize = 16)
            ax[i,j].ticklabel_format(style = "sci")
            ax[i,j].tick_params(labelsize = 16)
    fig.savefig(f"Images/Stokes_params/stokes_plot_0{obtained_file}-03.png")


    fig, ax = plt.subplots(2,2,figsize=figsize)

    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(0,max_height,1)+1, stokes[min_x_plot, min_z_plot, i*2+j], label = "generated stokes")
            ax[i,j].plot(np.arange(0,max_height,1)+1, original_stokes[min_x_plot, min_z_plot, i*2+j], label = "original stokes")
            ax[i,j].set_title("In minimum")
            ax[i,j].legend()
            ax[i,j].set_xlabel("height pixels")
            ax[i,j].set_ylabel(ylabels[i*2+j])
            ax[i,j].ticklabel_format(style = "sci")
    fig.savefig(f"Images/Stokes_params/stokes_plot_0{obtained_file}-11.png")

    fig, ax = plt.subplots(2,2,figsize=figsize)

    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(0,max_height,1)+1, stokes[min_x_plot, min_z_plot, i*2+j], label = "generated stokes")
            ax[i,j].set_title("In minimum")
            ax[i,j].legend()
            ax[i,j].set_xlabel("height pixels")
            ax[i,j].set_ylabel(ylabels[i*2+j])
            ax[i,j].ticklabel_format(style = "sci")
    fig.savefig(f"Images/Stokes_params/stokes_plot_0{obtained_file}-12.png")

    fig, ax = plt.subplots(2,2,figsize=figsize)

    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.arange(0,max_height,1)+1, original_stokes[min_x_plot, min_z_plot, i*2+j], label = "original stokes")
            ax[i,j].set_title("In minimum")
            ax[i,j].legend()
            ax[i,j].set_xlabel("height pixels")
            ax[i,j].set_ylabel(ylabels[i*2+j])
            ax[i,j].ticklabel_format(style = "sci")
    fig.savefig(f"Images/Stokes_params/stokes_plot_0{obtained_file}-13.png")


    fig, ax = plt.subplots(2,2,figsize=figsize)

    for i in range(2):
        for j in range(2):
            im_i = ax[i,j].imshow(stokes[:,:,i*2+j,wavelength], cmap="gist_gray")
            ax[i,j].scatter(max_x_plot, max_z_plot, label = "maximum", color = "r")
            ax[i,j].scatter(min_x_plot, min_z_plot, label = "minimun", color = "g")
            ax[i,j].legend()
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes('bottom', size='5%', pad=0.3)
            fig.colorbar(im_i, cax=cax, orientation="horizontal")
    fig.savefig(f"Images/Stokes_params/height_serie_plots_0{obtained_file}-2.png")
    
#


if __name__ == "__main__":
    main()