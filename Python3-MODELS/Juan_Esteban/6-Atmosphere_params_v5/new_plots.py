from this import d
from venv import create
import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_class

def main():
    obtained_file = int(input("File of the obtained atmosphere values: "))
    filename = "0"+str(obtained_file)
    atm_params = np.load(f"obtained_value-0{obtained_file}.npy")
    data = Data_class(create_scaler=False)
    original_atm_params = data.charge_atm_params(filename)
    height = 10
    max_height = len(atm_params[0,0,0,:])
    max_values = {}
    min_values = {}
    for i in range(4):
        max_values[i] = np.argwhere(atm_params[:,:,i,height]==np.max(atm_params[:,:,i,height]))
        min_values[i] = np.argwhere(atm_params[:,:,i,height]==np.min(atm_params[:,:,i,height]))
    
    fig, ax = plt.subplots(3,4,figsize=(40,40))
    for i in range(4):
        ax[0,i].plot(np.arange(0,max_height,1)+1, atm_params[max_values[i][0][0], max_values[i][0][1], i], label = "generated params")
        ax[0,i].plot(np.arange(0,max_height,1)+1, original_atm_params[max_values[i][0][0], max_values[i][0][1], i], label = "original params")
        ax[0,i].set_title("Height serie in a maximum")
        ax[0,i].legend()
        ax[1,i].plot(np.arange(0,max_height,1)+1, atm_params[max_values[i][0][0], max_values[i][0][1], i], label = "generated params")
        ax[1,i].plot(np.arange(0,max_height,1)+1, original_atm_params[max_values[i][0][0], max_values[i][0][1], i], label = "original params")
        ax[1,i].set_title("Height serie in a minimum")
        ax[1,i].legend()
        ax[2,i].imshow(atm_params[:,:,i,height], cmap="gist_gray")
        ax[2,i].scatter(max_values[i][0][0], max_values[i][0][1], label = "maximum", color = "r")
        ax[2,i].scatter(min_values[i][0][0], min_values[i][0][1], label = "minimun", color = "g")
        ax[2,i].legend()
    fig.savefig(f"Images/Stokes_params/height_serie_plots_0{obtained_file}.png")
    



if __name__ == "__main__":
    main()