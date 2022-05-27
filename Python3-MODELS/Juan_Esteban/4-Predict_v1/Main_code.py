import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import Sequential
import tensorflow as tf
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
#DATA TREATMENT MODULE
import data_functions as data_f
import nn_model 


## Parameters for the obtaining the data

#################################################################### MAIN FUNCTION ####################################################################
def main():
    path = '/mnt/scratch/juagudeloo/Total_MURAM_data/' #Juan Esteban path
    self_ptm = path
    self_filename = []
    file_interval = np.arange(53000, 223000, 50000)
    for i in range(len(file_interval)):
        if file_interval[i] < 100000:
            self_filename.append('0'+str(file_interval[i]))
        else:
            self_filename.append(str(file_interval[i]))
    self_filename.append('054000')
    print(self_filename)
    self_nx = 480
    self_ny = 256
    self_nz = 480
    iout, mbyy, mvyy, mtpr, mrho = data_f.charge_data(self_ptm, self_filename,
                                            self_nx, self_ny, self_nz)
    
    print(np.shape(iout))
    print(np.shape(mbyy))
    print(np.shape(mvyy))
    print(np.shape(mtpr))
    print(np.shape(mrho))
    data_f.plotting(iout, mbyy, mvyy, mrho, mtpr, self_nx, self_ny, self_nz)


    #training, testing and predicting division
    pr_n = 1 #predicting number
    model_lim = len(iout) - pr_n
    tr_n = int(model_lim*(3./4.)) #training number
    te_n = int(model_lim*(1./4.)) #testing number
    print(len(iout))
    print(f"tr_n = {tr_n}")
    print(f"te_n = {te_n}")
    print(f"pr_n = {pr_n}")
    
    print_shape = 0
    VERBOSE = 0
    data = []
    for i in range(len(mbyy)):
        data.append([mbyy[i], mvyy[i], mtpr[i], mrho[i]])
    labels = iout

    #DATA CLASSIFICATION
    TR_D, TR_L, TE_D, TE_L, PR_D, PR_L = data_f.data_classif(data, labels, 
                                                   self_nx, self_nz, tr_n, te_n, pr_n,
                                                    print_shape=1)

    IN_LS = np.array([4,256]) #input shape in input layer
    pr_BATCH_SIZE = int(len(PR_D)/10)
    print(pr_BATCH_SIZE)

    #TR_BATCH_SIZE = int(len(TR_D[:,1,2])/345600)
    TR_BATCH_SIZE = 2

    #PLOT INFORMATION
    title = [f"CNN 4 - TR BATCH SIZE = {TR_BATCH_SIZE}"]     
    dist_name = "nn_dense_models_dist.png"
    plot_var_values = True
    model = nn_model.NN_MODEL(self_nx, self_ny, self_nz, n_layers=4, epochs = 6)
    fitted = model.model_fitting(IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)
    print(f"PR_D shape = {np.shape(PR_D)}")
    model.predict_intensity(PR_D, pr_BATCH_SIZE)
    model.plot_dist(PR_L, title, dist_name)



if __name__ == "__main__":
    main()
