import os
import sys

import time
import datetime

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

#Own modules
sys.path.append("/girg/juagudeloo/Proyecto_DL-Trials/Inversion_model_1D/module")
from muram import MuRAM



def generate_new_data(ptm: str, 
                      model: nn.Module,
                      filename: str, 
                      batch_size: int,
                      vertical_comp: bool, 
                      ) -> None:
    """
    Function to generate new data.
    ------------------------------
    
    Args:
        model (nn.Module): Model to generate the data.
        ptm (str): Path to the Muram data.
        filename (str): Filename to generate the data.
        vertical_comp (bool): If the data has vertical component.
        batch_size (int): Batch size to generate the data.
    """
    
    muram = MuRAM(ptm = ptm, filenames = [''])
    atm_quant, stokes = muram.charge_quantities(filename=filename, vertical_comp = vertical_comp)

    ##############################################################################################
    # Charging the data for testing
    ##############################################################################################
    #Defining the agnostic device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    atm_quant_tensor = torch.from_numpy(atm_quant).to(device)
    atm_quant_tensor = torch.reshape(atm_quant_tensor, (480*480,20,6))
    stokes_tensor = torch.from_numpy(stokes).to(device)
    print(atm_quant_tensor.shape, stokes_tensor.shape)
    # stokes = torch.reshape(stokes,(stokes_s[0]*stokes_s[1], stokes_s[2], stokes_s[3]))

    validation_data = TensorDataset(stokes_tensor, atm_quant_tensor)
    validation_dataloader = DataLoader(validation_data,
            batch_size=batch_size,
            shuffle=False # don't necessarily have to shuffle the testing data
        )

    print(f"Length of validation dataloader: {len(validation_dataloader)} batches of {batch_size}")
    
    #########################################################################################################################
    # Number of parameters per model.
    #########################################################################################################################
    print("---------------------- Total number of parameters per model ----------------------\n")
    print(f" >> ", sum(p.numel() for p in model.parameters())/1e6, "M")
    
    start_time = time()
    generated_atm = torch.zeros((480*480,120))
    
    with torch.inference_mode():
        i = 0
        for X, y in validation_dataloader:
            # 1. Forward pass
            valid_pred = model.double()(X.double())
            generated_atm[i*80:(i+1)*80] = valid_pred
            i += 1
        generated_atm = np.reshape(generated_atm, (muram.nx, muram.nz, 20, atm_quant.shape[-1]))
        generated_atm = generated_atm.numpy()
    end_time = time()
    runtime = datetime.timedelta(seconds= (end_time - start_time))
    print(f"The generation of data with {device} took {runtime}")
    
    #####################################################################
    # Re scaling the quantities
    #####################################################################


    def re_scale_func(scaled_val, maxmin):
        max_val = maxmin[0]
        min_val = maxmin[1]
        return scaled_val*(max_val - min_val) + min_val


    N_quantities = atm_quant.shape[-1]

    for iatm in range(N_quantities):
        if iatm == 0:
            atm_maxmin = muram.phys_maxmin["T"]
        if iatm == 1:
            atm_maxmin = muram.phys_maxmin["Rho"]
        if iatm == 2:
            atm_maxmin = muram.phys_maxmin["B"]
        if iatm == 3:
            atm_maxmin = muram.phys_maxmin["V"]
        
        atm_quant[:,:,:,iatm] = re_scale_func(atm_quant[:,:,:,iatm], atm_maxmin)
        generated_atm[:,:,:,iatm] = re_scale_func(generated_atm[:,:,:,iatm], atm_maxmin)
        
    return stokes, atm_quant, generated_atm

titles = [r"$T$", r"$\rho$", r"$B_{q}$", r"$B_{u}$", r"$B_{v}$", r"$v_{LOS}$"]
tau = np.linspace(1,-3,20)
images_out = "Results/Images/"

def plot_pixel(filename:str,
               stokes: np.ndarray,
                atm_quant: np.ndarray,
                generated_atm: np.ndarray, 
                ix: int, iy: int) -> None:
    
    pixel_out = "Results/Images/pixel/"
    
    stokes_for_plot = np.reshape(stokes, (480,480,36,4))
    N_quantities = atm_quant.shape[-1]
    tau = np.linspace(1,-3,20)
    
    fig, ax = plt.subplots(1,1+N_quantities,figsize = (4*(1+N_quantities),4*1))
    
    ax[0].imshow(stokes_for_plot[:,:,0,0])
    ax[0].scatter(ix,iy, c = "orange")
    for j in range(1,1+N_quantities):
        ax[j].plot(tau, generated_atm[ix,iy,::-1,j], label = "generated")
        ax[j].plot(tau, atm_quant[ix,iy,::-1,j], label = "original")
        ax[j].set_title(titles[j])
        ax[j].legend()
    fig.tight_layout()
    
    if not os.path.exists(pixel_out):
        os.makedirs(pixel_out)
    fig.savefig(pixel_out+filename+f"pixel_{ix}_{iy}.png")
    
def plot_corr_diff_OD(filename: str,
                      atm_quant: np.ndarray,
                      generated_atm: np.ndarray,
                      iheight: int) -> None:
    
    N_quantities = atm_quant.shape[-1]
    
    fig, ax = plt.subplots(1,N_quantities,figsize = (N_quantities*5,1))
    for j in range(N_quantities):
        ax[j].scatter(atm_quant[:,:,iheight,j].flatten(), generated_atm[:,:,iheight,j].flatten(), s=5, alpha=0.1)
        max_value = np.max(np.array([np.max(generated_atm[:,:,iheight,j].flatten()),
                                                    np.max(atm_quant[:,:,iheight,j].flatten())]))
        min_value = np.max(np.array([np.min(generated_atm[:,:,iheight,j].flatten()),
                                                    np.min(atm_quant[:,:,iheight,j].flatten())]))
        max_y = np.max(generated_atm[:,:,iheight,j].flatten())
        max_x = np.max(atm_quant[:,:,iheight,j].flatten())
        min_y = np.min(generated_atm[:,:,iheight,j].flatten())
        min_x = np.min(atm_quant[:,:,iheight,j].flatten())
        pearson = pearsonr(generated_atm[:,:,iheight,j].flatten(),atm_quant[:,:,iheight,j].flatten())[0]
        ax[j].plot(np.linspace(min_value,max_value), np.linspace(min_value,max_value), "r--")
        ax[j].set_title(titles[j]+f" - OD = {np.linspace(-3,1,20)[iheight]:.2f} - pearson = {pearson:.2f}", fontsize=14)
        ax[j].sconstrainedet_xlim(min_x, max_x)
        ax[j].set_ylim(min_y, max_y)
    fig.tight_layout()
    fig.text(0.5, -0.02, 'Generated', ha='center',fontsize=14)
    fig.text(-0.02, 0.5, 'Original', va='center', rotation='vertical',fontsize=14)
    
    corr_out = images_out + "correlation/"
    if not os.path.exists(corr_out):
        os.makedirs(corr_out)
    fig.savefig(corr_out+filename+"_"+str(iheight)+".png")
    
def all_depth_error(filename: str,
                    atm_quant: np.ndarray,
                    generated_atm: np.ndarray) -> None:
    
    rrmse_atm_quant = np.reshape(atm_quant, (480*480, 20, atm_quant.shape[-1]))
    rrmse_generated_atm = np.reshape(generated_atm, (480*480, 20, generated_atm.shape[-1]))

    N_bins = 12

    t = 0
    atm = 2
    B_bins = np.linspace(np.min(rrmse_atm_quant[:,t,atm]), np.max(rrmse_atm_quant[:,t,atm]), N_bins+1)
    
    Bt_original = rrmse_atm_quant[:,t,atm]
    Bt_generated = rrmse_generated_atm[:,t,atm]

    min_idx = np.argwhere(Bt_original < B_bins[0+1])
    Bt_original_binned = np.squeeze(Bt_original[min_idx])
    Bt_generate_binned = np.squeeze(Bt_generated[min_idx])
    max_idx = np.argwhere(Bt_original_binned > B_bins[0])
    Bt_original_binned = np.squeeze(Bt_original_binned[max_idx])
    Bt_generate_binned = np.squeeze(Bt_generate_binned[max_idx])
    
    Atm_bins = {0:{}, #T
                1:{}, #rho
                2:{}, #Bx
                3:{}, #By
                4:{}, #Bz
                5:{}} #vy

    for t in range(len(tau)):
        for atm in range(rrmse_atm_quant.shape[-1]):
            Atm_bins[atm][t] = np.linspace(np.min(rrmse_atm_quant[:,t,atm]), np.max(rrmse_atm_quant[:,t,atm]), N_bins+1)

    binned_atm_data = {"original":{},
                    "validation":{}}

    binned_atm_data["original"] = {0:{}, #T
                1:{}, #rho
                2:{}, #Bx
                3:{}, #By
                4:{}, #Bz
                5:{}} #vy

    binned_atm_data["validation"] = {0:{}, #T
                1:{}, #rho
                2:{}, #Bx
                3:{}, #By
                4:{}, #Bz
                5:{}} #vy


    for t in range(len(tau)):
        for atm in range(rrmse_atm_quant.shape[-1]):
            binned_atm_data["original"][atm][t] = {}
            binned_atm_data["validation"][atm][t] = {}
            
    for t in range(len(tau)):
        for atm in range(rrmse_atm_quant.shape[-1]):      
            for i_bin in range(N_bins):
                binned_atm_data["original"][atm][t][i_bin] = []
                binned_atm_data["validation"][atm][t][i_bin] = []

    for i_bin in range(N_bins):
        for t in range(len(tau)):
            for atm in range(rrmse_atm_quant.shape[-1]):
                atm_tau = rrmse_atm_quant[:,t,atm]
                min_bin = Atm_bins[atm][t][i_bin]
                max_bin = Atm_bins[atm][t][i_bin+1]
                
                max_idx = np.argwhere(atm_tau < max_bin)
                max_idx = np.squeeze(max_idx)   
                binned_atm_data["original"][atm][t][i_bin] = rrmse_atm_quant[max_idx,t,atm]
                binned_atm_data["validation"][atm][t][i_bin] = rrmse_generated_atm[max_idx,t,atm]
                
                min_idx = np.argwhere(binned_atm_data["original"][atm][t][i_bin] > min_bin)
                min_idx = np.squeeze(min_idx)   
                binned_atm_data["original"][atm][t][i_bin] = binned_atm_data["original"][atm][t][i_bin][min_idx]
                binned_atm_data["validation"][atm][t][i_bin] =  binned_atm_data["validation"][atm][t][i_bin][min_idx]
                



    rmse_t = {0:{}, #T
                1:{}, #rho
                2:{}, #Bx
                3:{}, #By
                4:{}, #Bz
                5:{}} #vy
    for t in range(len(tau)):
        for atm in range(rrmse_atm_quant.shape[-1]):
            rmse_t[atm][t] = np.zeros((N_bins,))
            
    

    def rrmse(y_actual, y_predicted):
        rms = mean_squared_error(y_actual, y_predicted, squared=False)
        rel_root_mse = rms/np.abs(np.mean(y_actual))
        return rel_root_mse
    
    for t in tqdm(range(len(tau))):
        for atm in range(rrmse_atm_quant.shape[-1]):
            for i_bin in range(N_bins):
                try:
                    original = binned_atm_data["original"][atm][t][i_bin]
                    generated = binned_atm_data["validation"][atm][t][i_bin]
                    rmse_t[atm][t][i_bin] = rrmse(original, generated)
                except:
                    rmse_t[atm][t][i_bin] = np.nan
                    
    N_OD = len(tau)
    N_atm = rrmse_atm_quant.shape[-1]
    bin_means = np.zeros((N_bins,))

    fig, ax = plt.subplots(N_OD, N_atm, figsize = (4*N_atm, 4*N_OD))
    for t in range(len(tau)):
        for atm in range(rrmse_atm_quant.shape[-1]):
            for i_bin in range(N_bins):
                bin_means[i_bin] = (Atm_bins[atm][t][i_bin] + Atm_bins[atm][t][i_bin+1])/2
            min_xlim = np.min(rrmse_atm_quant[:,:,atm])
            max_xlim = np.max(rrmse_atm_quant[:,:,atm])
            ax[t,atm].plot(bin_means, rmse_t[atm][t], color = "r")
    #         ax[t,atm].plot(np.linspace(min_xlim, max_xlim),np.zeros_like(np.linspace(min_xlim, max_xlim)), "--k")
            ax[t,atm].scatter(bin_means, rmse_t[atm][t], color = "r")
            ax[t,atm].set_ylim(0,1)
            ax[t,atm].ticklabel_format(scilimits = (-3,3))
    #         ax[t,atm].set_xlim(min_xlim, max_xlim)
            ax[t,atm].set_title(titles[atm] + f" OD {tau[t]:0.2f}")

    fig.tight_layout()
    fig.savefig("rrmse_dif_OD.png", transparent = False)








