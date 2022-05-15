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
    self_filename = '154000'
    self_nx = 480
    self_ny = 256
    self_nz = 480
    iout, mbyy, mvyy, mtpr, mrho = data_f.charge_data(self_ptm, self_filename,
                                            self_nx, self_ny, self_nz)
    data_f.plotting(iout, mbyy, mvyy, mrho, mtpr, self_nx, self_ny, self_nz)


    #training, testing and predicting division
    tr_div_nx = 300
    tr_div_nz = 480

    te_div_nx = 100
    te_div_nz = 480

    pr_div_nx = 80
    pr_div_nz = 480

    print_shape = 0
    VERBOSE = 0

    data = np.array([mbyy, mvyy, mtpr, mrho])
    labels = iout

    #DATA CLASSIFICATION
    TR_D, TR_L, TE_D, TE_L, PR_D, PR_L = data_f.data_classif(data, labels, 
                                                    tr_div_nx, tr_div_nz, 
                                                    te_div_nx, te_div_nz, 
                                                    pr_div_nx, pr_div_nz, 
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

    #TESTING LIST VALUES
    #optimizing functions
    lr = 0.001
    opt_func = [tf.keras.optimizers.SGD(learning_rate=lr),
            tf.keras.optimizers.RMSprop(learning_rate=lr),
            tf.keras.optimizers.Adam(learning_rate=lr),
            tf.keras.optimizers.Adadelta(learning_rate=lr),
            tf.keras.optimizers.Adagrad(learning_rate=lr)]

    #PLOT INFORMATION
    titles = ['SGD',
                  'RMSprop',
                  'Adam',
                  'Adadelta',
                  'Adagrad']
                            
    dist_name = "optimizing_dist_1.png"
    error_fig_name = "optimizing_error_1.png"
    plot_var_values = True
    ##### NEURAL NETWORK TYPE VARIATIONS
    #dense models

  

    models, metrics, history = test.test_opt_func(opt_func, IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)

    intensity_pred = test.predict_intensity(models, PR_D, pr_BATCH_SIZE, pr_div_nx, pr_div_nz)
    var_values = range(len(models))
    data_f.plot_dist(intensity_pred, history, metrics,
                        titles, PR_L, dist_name, 
                        error_fig_name, var_values, plot_var_values)



if __name__ == "__main__":
    main()