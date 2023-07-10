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
    opt_mags_interp = {}

    N = 50
    tau = np.linspace(-5, 0.5, N)
    opt_mags = [np.zeros(N), #mbyy
                np.zeros(N), #mvyy
                np.zeros(N), #log(mrho)
                np.zeros(N)] #mtpr
    ix, iz = 200,200
    for i in range(4):
        opt_mags_interp[mags_names[i]] = interp1d(opt_depth[ix,:,iz], muram.atm_params[ix,iz,:,i])
        opt_mags[i][:] = opt_mags_interp[mags_names[i]](tau)
    fig, ax = plt.subplots(figsize = (30,7))
    for i in range(4):
        ax[i].plot(tau, opt_mags[:])
    fig.savefig("optical_depth_height_mapping-"+filename+".pdf")


    opt_mags = [np.zeros((480,480)), #mbyy
                np.zeros((480,480)), #mvyy
                np.zeros((480,480)), #log(mrho)
                np.zeros((480,480))] #mtpr

    
    tau = 0 #value of optical depth for the remapping

    for ix in range(480):
        for iz in range(480):
            for i in range(4):
                opt_mags_interp[mags_names[i]] = interp1d(opt_depth[ix,:,iz], muram.atm_params[ix,iz,:,i])
                opt_mags[i][ix,iz] = opt_mags_interp[mags_names[i]](tau)

    fig, ax = plt.subplots(1,4,figsize=(30,9))
    for i in range(4):
        im = ax[i].imshow(opt_mags[i])
        ax[i].set_title(mags_names[i])
        fig.colorbar(im, ax = ax[i])

    fig.savefig("optical_depth_2D_mapping-"+filename+".pdf")

    






if __name__ == "__main__":
    main()