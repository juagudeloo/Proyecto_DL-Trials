from path import path_UIS
from train_generate.data_class import DataClass
import matplotlib.pyplot as plt 
import numpy as np

def main():
    ptm = path_UIS()
    muram = DataClass(ptm)

    filename = "087000"
    stokes_87 = muram.charge_stokes_params(filename, scale = False)[:,:,0,0]
    threshold_87 = (np.max(stokes_87)-np.min(stokes_87))/2.5 + np.min(stokes_87)
    gran_mask_87 = np.ma.masked_where(stokes_87 > threshold_87, stokes_87).mask
    intergran_mask_87 = np.ma.masked_where(stokes_87 <= threshold_87, stokes_87).mask

    filename = "175000"
    stokes_175 = muram.charge_stokes_params(filename, scale = False)[:,:,0,0]
    threshold_175 = (np.max(stokes_175)-np.min(stokes_175))/2.5 + np.min(stokes_175)
    gran_mask_175 = np.ma.masked_where(stokes_175 > threshold_175, stokes_175).mask
    intergran_mask_175 = np.ma.masked_where(stokes_175 <= threshold_175, stokes_175).mask

    fig, ax = plt.subplots(2,2,figsize = (12,5))
    ax[0,0].imshow(stokes_87)
    ax[0,0].imshow(gran_mask_87, alpha = 0.6, label = "granular")
    ax[0,0].imshow(intergran_mask_87, alpha = 0.6, label = "intergranular")
    ax[0,0].legend()
    ax[1,0]..imshow(stokes_87)
    
    ax[0,1].imshow(stokes_175)
    ax[0,1].imshow(gran_mask_175, alpha = 0.6, label = "granular")
    ax[0,1].imshow(intergran_mask_175, alpha = 0.6, label = "intergranular")
    ax[0,1].legend()
    ax[1,1].imshow(stokes_175)

    fig.savefig("int_gran_zones.pdf")



if __name__ == "__main__":
    main()