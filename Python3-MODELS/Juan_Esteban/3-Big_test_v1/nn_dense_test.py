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
import testing_functions as test


## Parameters for the obtaining the data

#################################################################### MAIN FUNCTION ####################################################################
def main():
    path = '/mnt/scratch/juagudeloo/Total_MURAM_data/' #Juan Esteban path
    self_ptm = path
    self_filename = []
    file_interval = np.arange(54000, 223000, 10000)
    for i in range(len(file_interval)):
        if file_interval[i] < 100000:
            self_filename.append('0'+str(file_interval[i]))
        else:
            self_filename.append(str(file_interval[i]))
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
    
    print_shape = 0
    VERBOSE = 0

    data = np.array([mbyy, mvyy, mtpr, mrho])
    labels = iout

    #DATA CLASSIFICATION
    TR_D, TR_L, TE_D, TE_L, PR_D, PR_L = data_f.data_classif(data, labels, 
                                                   self_nx, self_nz, tr_n, te_n, pr_n,
                                                    print_shape=1)

    IN_LS = np.array([4,256]) #input shape in input layer
    pr_BATCH_SIZE = int(len(PR_D)/10)
    print(pr_BATCH_SIZE)

    TR_BATCH_SIZE = int(len(TR_D[:,1,2])/1)

    #MODELS DATA DICTIONARY
    models = []
    history = []
    metrics = []
    intensity_pred = []

    #PLOT INFORMATION
    titles = ['Dense 1','Dense 2','Dense 3','Dense4']
                            
    dist_name = "nn_dense_models_dist.png"
    error_fig_name = "nn_dense_models_error.png"
    plot_var_values = True
    ##### NEURAL NETWORK TYPE VARIATIONS
    #dense models

  

    models, metrics, history = test.test_dense_models(IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)

    intensity_pred = test.predict_intensity(models, PR_D, pr_BATCH_SIZE, pr_div_nx, pr_div_nz)
    var_values = range(len(models))
    data_f.plot_dist(intensity_pred, history, metrics,
                        titles, PR_L, dist_name, 
                        error_fig_name, var_values, plot_var_values)



if __name__ == "__main__":
    main()
