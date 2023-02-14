import os
import numpy as np
import matplotlib.pyplot as plt
from train_generate.data_class import DataClass

path = "/girg/juagudeloo/MURAM_data/"
filename = "134000"

d_class = DataClass(path, light_type = "Stokes_params")

atm_params = d_class.charge_atm_params(filename)
stokes_params = d_class.charge_stokes_paramas(filename)

del d_class

fig, ax = plt.subplots(4,4, figsize = (50,50))
for i in range(4):
    ax[0, i].imshow(atm_params[:,:,10,i])
for i in range(4):
    ax[1, i].plot(atm_params[10,10,:,i])
for i in range(4):
    ax[2, i].imshow(stokes_params[:,:,10,i])
for i in range(4):
    ax[3, i].plot(atm_params[10,10,:,i])
dir_name = "check_data_directory"
os.mkdir(dir_name)
fig.savefig(dir_name+"/"+filename+"checking.png")