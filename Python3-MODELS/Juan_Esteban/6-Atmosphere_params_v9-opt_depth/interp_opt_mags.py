import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from train_generate.data_class import DataClass
from path import path_UIS

def main():
    ptm = path_UIS()
    muram = DataClass(ptm, lower_boundary = 0)

    filename = "175000"
    opt_depth = np.load("optical_depth_"+filename+".npy")
    muram.charge_atm_params(filename, scale = False)
    


    mags_names = ["By_opt", "Vy_opt", "log_rho_opt", "T_opt"]
    opt_mags = [np.zeros((480,480)), #mbyy
                np.zeros((480,480)), #mvyy
                np.zeros((480,480)), #log(mrho)
                np.zeros((480,480))] #mtpr

    opt_mags_interp = {}
    tau = 1 #value of optical depth for the remapping

    for ix in range(480):
        for iz in range(480):
            for i in range(4):
                opt_mags_interp[mags_names[i]] = interp1d(opt_depth[ix,:,iz], muram.atm_params[ix,iz,:,i])
                opt_mags[i][ix,iz] = opt_mags_interp[mags_names[i]](tau)

    fig, ax = plt.subplots(1,4,figsize=(30,9))
    for i in range(4):
        ax[i].imshow(opt_mags[i])
        ax[i].set_title(mags_names[i])

    fig.savefig("optical_depth_mapping-"+filename+".pdf")





if __name__ == "__main__":
    main()