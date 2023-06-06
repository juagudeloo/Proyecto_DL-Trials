import matplotlib.pyplot as plt
import numpy as np
from train_generate.data_class import DataClass
from path import path_UIS
from sklearn.metrics import r2_score

def main():
    ptm = path_UIS()
    muram_data = DataClass(ptm, lower_boundary=180)
    filename = "175000"

    muram_data.charge_atm_params(filename, scale = False)
    muram_data.charge_stokes_params(filename, scale = False)
    
    nfile = str(175000)
    obtained_file = "obtained_value-"+nfile+".npy"
    atm_ptm = "atm_NN_model/Predicted_values/Stokes params/"
    stokes_ptm = "light_NN_model/Predicted_values/Stokes params/"
    pred_atm = np.load(atm_ptm+obtained_file)
    pred_stokes = np.load(stokes_ptm+obtained_file)

    #Mask for stablishing the R^2
    threshold = (np.max(muram_data.profs[:,:,0,0])-np.min(muram_data.profs[:,:,0,0]))/2.5 + np.min(muram_data.profs[:,:,0,0])
    intergran_mask = np.ma.masked_where(muram_data.profs[:,:,0,0] > threshold, muram_data.profs[:,:,0,0]).mask
    gran_mask = np.ma.masked_where(muram_data.profs[:,:,0,0] <= threshold, muram_data.profs[:,:,0,0]).mask

    #################################
    #PHYSICAL MAGNITUDES
    #################################

    phys_mags_tit = ["Magnetic Field LOS", "Velocity LOS", "Density", "Temperature"]
    #R^2 for intergranular zones
    for i in range(4):
        print(phys_mags_tit[i], r2_score(muram_data.atm_params[intergran_mask,:,i], pred_atm[intergran_mask,:,i]))

    #R^2 for granular zones
    for i in range(4):
        print(phys_mags_tit[i], r2_score(muram_data.atm_params[gran_mask,:,i], pred_atm[gran_mask,:,i]))

    #R^2 for all the data
    
    for i in range(4):
        print(muram_data.atm_params[:,:,:,i].shape)
        print(pred_atm[:,:,:,i].shape)
        print(phys_mags_tit[i], r2_score(np.ravel(muram_data.atm_params[:,:,:,i]), np.ravel(pred_atm[:,:,:,i])))

    #################################
    #STOKES
    #################################

    stokes_tit = ["I", "U", "Q", "V"]
    #R^2 for intergranular zones
    for i in range(4):
        print(stokes_tit[i], r2_score(muram_data.profs[intergran_mask,:,i], pred_stokes[intergran_mask,:,i]))

    #R^2 for granular zones
    for i in range(4):
        print(stokes_tit[i], r2_score(muram_data.profs[gran_mask,:,i], pred_stokes[gran_mask,:,i]))

    #R^2 for all the data
    for i in range(4):
        print(stokes_tit[i], r2_score(np.ravel(muram_data.profs[:,:,:,i]), np.ravel(pred_stokes[:,:,:,i])))
    

    #ix = 90
    #iz = 15
    #iy = 0
    #ilam = 0
    #N_stokes = muram_data.nlam
    #n_atm_param = 2
#
    #fig, ax = plt.subplots(4,4,figsize = (30,14))
    #for i in range(4):
    #    ax[0,i].plot(np.arange(0,76,1)+1, pred_atm[ix,iz,:,i], c = "k", label = "original")
    #    ax[0,i].plot(np.arange(0,76,1)+1, muram_data.atm_params[ix,iz,:,i], c = "r", label = "generated")
    #    ax[0,i].set_title(f"R² = {r2_score(pred_atm[ix,iz,:,i], muram_data.atm_params[ix,iz,:,i])}")
    #    ax[0,i].legend()
    #    ax[1,i].imshow(muram_data.atm_params[:,:,iy,i], cmap = "gist_gray")
    #    ax[1,i].scatter(ix,iz,c="r",s=20)
    #    ax[2,i].plot(np.arange(0,N_stokes,1)+1, muram_data.profs[ix,iz,:,i], c = "k", label = "original")
    #    ax[2,i].plot(np.arange(0,N_stokes,1)+1, pred_stokes[ix,iz,:,i], c = "r", label = "generated")
    #    ax[2,i].set_title(f"R² = {r2_score(pred_stokes[ix,iz,:,i], muram_data.profs[ix,iz,:,i])}")
    #    ax[2,i].legend()
    #    ax[2,i].scatter(ilam, muram_data.profs[ix,iz,ilam,i], c="orange")
    #    ax[3,i].imshow(muram_data.profs[:,:,ilam,i], cmap = "gist_gray")
    #    ax[3,i].scatter(ix,iz,c="r",s=20)

if __name__ == "__main__":
    main()