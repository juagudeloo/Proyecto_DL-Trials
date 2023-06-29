import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from train_generate.data_class import DataClass
from path import path_UIS

def main():
    ptm = path_UIS()
    muram = DataClass(ptm, lower_boundary = 0)

    ix = 200
    iz = 200

    filename = "175000"
    opt_depth = np.load("optical_depth_"+filename+".npy")

    muram.charge_atm_params(filename)

    mags = ["By_opt", "Vy_opt", "log_rho_opt", "T_opt"]
    opt_mags_interp = {}
    for i in range(4):
        opt_mags_interp[mags[i]] = interp1d(opt_depth[ix,:,iz], muram.atm_params[ix,:,iz,i])
    
    opt_grid = np.arange(-2,5,1)

    fig, ax = plt.subplots(1,4,figsize=(7,7))
    for i in range(4):
        ax[i].plot(opt_grid, opt_mags_interp[mags[i]](opt_grid))
        ax[i].set_title(mags[i])

    fig.savefig("optical_depth_mapping"+filename+".pdf")





if __name__ == "__main__":
    main()