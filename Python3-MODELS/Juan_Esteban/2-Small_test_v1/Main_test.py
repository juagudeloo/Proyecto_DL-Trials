
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

  #DICTIONARY NAMES
  names = ['Dense_layers', 'CNN_layers', 'opt_func', 'loss_func', 'learning_rate', 'batch_size']
  
  #MODELS DATA DICTIONARY
  models = {names[0]: [],
          names[1]: [],
          names[2]: [],
          names[3]: [],
          names[4]: [],
          names[5]: []}
  history = {names[0]: [],
          names[1]: [],
          names[2]: [],
          names[3]: [],
          names[4]: [],
          names[5]: []}
  metrics = {names[0]: [],
          names[1]: [],
          names[2]: [],
          names[3]: [],
          names[4]: [],
          names[5]: []}
  intensity_pred = {names[0]: [],
          names[1]: [],
          names[2]: [],
          names[3]: [],
          names[4]: [],
          names[5]: []}

  #TESTING LIST VALUES
  #optimizing functions
  lr = 0.001
  opt_func = [tf.keras.optimizers.SGD(learning_rate=lr),
            tf.keras.optimizers.RMSprop(learning_rate=lr),
            tf.keras.optimizers.Adam(learning_rate=lr),
            tf.keras.optimizers.Adadelta(learning_rate=lr),
            tf.keras.optimizers.Adagrad(learning_rate=lr),
            tf.keras.optimizers.Adamax(learning_rate=lr),
            tf.keras.optimizers.Nadam(learning_rate=lr),
            tf.keras.optimizers.Ftrl(learning_rate=lr)]
  #loss function 
  loss_func = [tf.keras.losses.BinaryCrossentropy(from_logits = True),
             tf.keras.losses.CategoricalCrossentropy(from_logits = True),
            #  tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             tf.keras.losses.Poisson(),
             tf.keras.losses.MeanSquaredError(),
             tf.keras.losses.MeanAbsoluteError(),
             tf.keras.losses.MeanAbsolutePercentageError(),
             tf.keras.losses.MeanSquaredLogarithmicError(),
             tf.keras.losses.CosineSimilarity()]
  loss_func_metrics = [tf.keras.metrics.BinaryCrossentropy(),
                    tf.keras.metrics.CategoricalCrossentropy(),
                    #  tf.keras.metrics.SparseCategoricalCrossentropy(),
                    tf.keras.metrics.Poisson(),
                    tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.MeanAbsolutePercentageError(),
                    tf.keras.metrics.MeanSquaredLogarithmicError(),
                    tf.keras.metrics.CosineSimilarity()]
  #learning rate
  lr_list = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
  #batch size
  batch_sizes = [int(len(TR_D[:,1,2])), int(len(TR_D[:,1,2])/10), 
               int(len(TR_D[:,1,2])/100), int(len(TR_D[:,1,2])/1000), 
               int(len(TR_D[:,1,2])/14400)]

  #PLOT INFORMATION
  titles = {names[0]: ['Dense 1','Dense 2','Dense 3','Dense4'],
                names[1]: ['CNN 1', 'CNN 2', 'CNN 3', 'CNN 4'],
                names[2]: ['SGD',
                  'RMSprop',
                  'Adam',
                  'Adadelta',
                  'Adagrad',
                  'Adamax',
                  'Nadam',
                  'Ftrl'],
                names[3]: ['BinaryCrossentropy', 
                  'CategoricalCrossentropy',
                  'Poisson',
                  'MeanSquaredErros',
                  'MeanAbsoluteError',
                  'MeanAbsolutePercentageError',
                  'MeanSquaredLogarithmicError',
                  'CosineSimilarity'],
                names[4]: [f'CNN 4 - lr = {lr_list[0]}', 
                           f'CNN 4 - lr = {lr_list[1]}',
                           f'CNN 4 - lr = {lr_list[2]}',
                           f'CNN 4 - lr = {lr_list[3]}',
                           f'CNN 4 - lr = {lr_list[4]}',
                           f'CNN 4 - lr = {lr_list[5]}',
                           f'CNN 4 - lr = {lr_list[6]}'],
                names[5]: [f'CNN 4 - lr = {lr_list[0]}', 
                f'CNN 4 - batch size = {batch_sizes[1]}',
                f'CNN 4 - batch size= {batch_sizes[2]}',
                f'CNN 4 - batch size = {batch_sizes[3]}',
                f'CNN 4 - batch size = {batch_sizes[4]}']}
                           
  dist_name = {names[0]: "nn_dense_models_dist.png",
          names[1]: "nn_conv_models_dist.png",
          names[2]: "optimizing_dist.png",
          names[3]: "loss_function_dist.png",
          names[4]: "learning_rate_models_dist.png",
          names[5]: "batch_size_dist.png"}
  error_fig_name = {names[0]: "nn_dense_models_error.png",
          names[1]: "nn_conv_models_error.png",
          names[2]: "optimizing_error.png",
          names[3]: "loss_function_error.png",
          names[4]: "learning_rate_models_error.png",
          names[5]: "batch_size_error.png"}
  plot_var_values = {names[0]: False,
          names[1]: False,
          names[2]: False,
          names[3]: False,
          names[4]: True,
          names[5]: True}
  ##### NEURAL NETWORK TYPE VARIATIONS
  #dense models

  for model_type in names:
    if model_type == names[0]:
      var_values = 0
      models[model_type], metrics[model_type], history[model_type] = test.test_dense_models(IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)
    if model_type == names[1]:
      var_values = 0
      models[model_type], metrics[model_type], history[model_type] = test.test_conv_models(IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)
    if model_type == names[2]:
      var_values = 0
      models[model_type], metrics[model_type], history[model_type] = test.test_opt_func(opt_func, IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)
    if model_type == names[3]:
      var_values = 0
      models[model_type], metrics[model_type], history[model_type] = test.test_loss_func(loss_func, loss_func_metrics, IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)
    if model_type == names[4]:
      models[model_type], metrics[model_type], history[model_type] = test.test_lr(lr_list, IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)
      var_values = lr_list
    if model_type == names[5]:
      var_values = batch_sizes
      models[model_type], metrics[model_type], history[model_type] = test.test_batch_size(batch_sizes, IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)

    intensity_pred[model_type] = test.predict_intensity(models[model_type], PR_D, pr_BATCH_SIZE, pr_div_nx, pr_div_nz)
    
    data_f.plot_dist(intensity_pred[model_type], history[model_type], metrics[model_type],
                      titles[model_type], PR_L, dist_name[model_type], 
                      error_fig_name[model_type], var_values, plot_var_values[model_type])



if __name__ == "__main__":
    main()
