import matplotlib.pyplot as plt
import numpy as np

def main():
    obtained_file = int(input("File of the obtained atmosphere values: "))
    atm_params = np.load(f"obtained_value-0{obtained_file}000.npy")
    height = 10
    max_height = len(atm_params[0,0,0,:])
    max_values = {}
    min_values = {}
    for i in range(4):
        max_values[i] = np.argwhere(atm_params[:,:,i,height]==np.max(atm_params[:,:,i,height]))
        min_values[i] = np.argwhere(atm_params[:,:,i,height]==np.min(atm_params[:,:,i,height]))
    
    for i in range(4):
        print(atm_params[max_values[i][0][0], max_values[i][0][1], i, height])

    fig, ax = plt.subplots(3,4,figsize=(40,40))
    for i in range(4):
        ax[0,i].plot(np.arange(0,max_height,1)+1, atm_params[max_values[i][0][0], max_values[i][0][1], i])
        ax[0,i].title("Height serie in a maximum")
        ax[1,i].plot(np.arange(0,max_height,1)+1, atm_params[max_values[i][0][0], max_values[i][0][1], i])
        ax[1,i].title("Height serie in a minimum")
        ax[2,i].imshow(atm_params[:,:,i,height])
        ax[2,i].scatter(max_values[i][0][0], max_values[i][0][1], label = "maximum", color = "r")
        ax[2,i].scatter(min_values[i][0][0], min_values[i][0][1], label = "minimun", color = "g")
        ax[2,i].legend()
    fig.save("Images/Stokes_params/height_serie_plots.png")
    



if __name__ == "__main__":
    main()