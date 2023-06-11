from train_generate.data_class import DataClass
import numpy as np
from path import path_UIS
import matplotlib.pyplot as plt

def main():
    plt.rcParams["font.size"] = 20
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({
        "font.family": "roman-serif",
        "font.sans-serif": "Helvetica",
    })
    plt.rcParams.update({'pgf.preamble': r'\usepackage{amsmath}'})

    ptm = path_UIS()
    muram= DataClass(ptm = ptm)
    filename = "175000"

    #Charging the original data
    stokes_origin = muram.charge_stokes_params(filename=filename, scale = False)
    atm_origin = muram.charge_atm_params(filename=filename, scale = False)

    #Charging the obtained data
    stokes_obt = np.load("light_NN_model/Predicted_values/Stokes params/obtained_value-175000.npy")
    atm_obt = np.load("atm_NN_model/Predicted_values/Stokes params/obtained_value-175000.npy")

    #Granular and intergranular positions
    ix = [200,200]
    iz = [20, 90] #0 - intergranular, 1 - granular

    #Plots
    """
    =================================================================================
    ==================================== STOKES =====================================
    =================================================================================
    """

    plot_stokes(ix, iz, stokes_origin, stokes_obt)

    """
    =================================================================================
    ===================================== ATM =======================================
    =================================================================================
    """

    plot_atm(ix, iz, atm_origin, atm_obt, stokes_origin[:,:,0,0])

def plot_stokes(ix:list, iz:list, origin, obtained, ilam = 0, ptm = "tdg_images/Stokes_params"): 

    cm = 1/2.54  # centimeters in inches

    #Intensity reference points for granular and intergranular zones
    fig, ax = plt.subplots((5*cm,5*cm))
    ax.imshow(origin[:,:,ilam,0])
    fig.save(ptm+"tdg_stokes.pdf")

    #Stokes plot
    titles = ["granular", "intergranular"]
    y_labels = [r"$I$"+"[u.a]", r"$Q$", r"$U$", r"$V$"]
    x_label = "wavelength, "+r"$\lambda\,[\AA]$"
    lam = np.arange(6300, 6300+10*300, 10)

    fig, ax = plt.subplots(1,4, figsize = (22*cm,5*cm))
    for j in range(2):
        for i in range(4):
            ax[j,i].plot(lam, origin[ix[j],iz[j],:,i])
            ax[j,i].plot(lam, obtained[ix[j],iz[j],:,i])
            ax[j,i].set_title(titles[j])
            ax[j,i].set_ylabel(y_labels[i])
            ax[j,i].set_xlabel(x_label)
        

    fig.save(ptm+"tdg_stokes.pdf")
    
def plot_atm(ix, iz, origin, obtained, I_reference, ptm = "tdg_images/Atm_params"):

    cm = 1/2.54  # centimeters in inches

    #Intensity reference points for granular and intergranular zones
    fig, ax = plt.subplots(1,4,figsize = (5*cm,5*cm))
    ax.imshow(I_reference)
    fig.save(ptm+"tdg_stokes.pdf")

    fig, ax = plt.subplots(1,4, figsize = (22*cm,5*cm))
    for i in range(4):
        ax[0,i].plot(origin[ix[0],iz[0],:,i])
        ax[0,i].plot(obtained[ix[0],iz[0],:,i])
        ax[1,i].plot(origin[ix[1],iz[1],:,i])
        ax[1,i].plot(obtained[ix[1],iz[1],:,i])

    #Atm plot
    titles = ["granular", "intergranular"]
    y_labels = [r"$B_\text{LOS}$"+"[G]", r"$v_\text{LOS}$", r"$\log{\rho}$"+"[$\log{\text{g/cm}}]", r"$T$"+"[K]"]
    x_label = "height [km]"
    Y = np.arange(0,76,1)*10*(-1)

    fig, ax = plt.subplots(1,4, figsize = (22*cm,5*cm))
    for j in range(2):
        for i in range(4):
            ax[j,i].plot(Y,origin[ix[j],iz[j],:,i])
            ax[j,i].plot(Y,obtained[ix[j],iz[j],:,i])
            ax[j,i].set_title(titles[j])
            ax[j,i].set_ylabel(y_labels[i])
            ax[j,i].set_xlabel(x_label)
    
    fig.save(ptm+"tdg_stokes.pdf")
    


if __name__ == "__main__":
    main()
