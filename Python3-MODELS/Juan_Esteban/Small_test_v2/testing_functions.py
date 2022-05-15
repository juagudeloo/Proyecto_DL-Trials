import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

#PREDICTING FUNCTION
def predict_intensity(models, PR_D:float, pr_BATCH_SIZE:float, pr_div_nx:float, pr_div_nz:float):
  print("*predicting intensity...*")
  row_len = len(models)
  intensity_out = []
  for i in range(len(models)):
    intensity_out.append(models[i].predict(PR_D, batch_size=pr_BATCH_SIZE, verbose=1))
    intensity_out[i] = intensity_out[i].reshape(pr_div_nx, pr_div_nz)
    print(np.shape(intensity_out[i]))

  print("*intensity_done*\n")
  return intensity_out

#NN MODEL TESTING
def test_dense_models(IN_LS, TR_D, TR_L, TR_BATCH_SIZE:float, TE_D, TE_L):
    models = []
    var_metrics = []
    history =[]
    N_models = 4
    opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
    for i in range(N_models):
        models.append(model_dense_layers(IN_LS, n_layers = 1+i))

    for i in range(N_models):
        models[i].compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        history.append(models[i].fit(TR_D, TR_L, epochs=8, batch_size=TR_BATCH_SIZE, verbose=1)) #I think becasuse of the size of the batch     

        start_time = time.time()
        #FITTING
        models[i].fit(TR_D, TR_L, epochs=1, batch_size=TR_BATCH_SIZE, verbose=1) #I think becasuse of the size of the batch     
                                                                            # it is charging 28800 number of data per epoch.
        #TESTING
        print('*TEST*')
        metrics = models[i].evaluate(TE_D, TE_L)
        var_metrics = np.append(var_metrics, metrics[1])
        print("%s: %2.f%%" % (models[i].metrics_names[1], metrics[1]*100))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')
    print('\n DENSE NEURAL TYPES DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    return models, var_metrics, history
def test_conv_models(IN_LS, TR_D, TR_L, TR_BATCH_SIZE:float, TE_D, TE_L):
    models = []
    var_metrics = []
    history =[]
    N_models = 4
    opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
    for i in range(N_models):
        models.append(model_dense_layers(IN_LS, n_layers = 1+i))

    for i in range(N_models):
        models[i].compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        history.append(models[i].fit(TR_D, TR_L, epochs=8, batch_size=TR_BATCH_SIZE, verbose=1)) #I think becasuse of the size of the batch     

        start_time = time.time()
        #FITTING
        models[i].fit(TR_D, TR_L, epochs=1, batch_size=TR_BATCH_SIZE, verbose=1) #I think becasuse of the size of the batch     
                                                                            # it is charging 28800 number of data per epoch.
        #TESTING
        print('*TEST*')
        metrics = models[i].evaluate(TE_D, TE_L)
        var_metrics = np.append(var_metrics, metrics[1])
        print("%s: %2.f%%" % (models[i].metrics_names[1], metrics[1]*100))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')
    print('\n CONV NEURAL TYPES DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    return models, var_metrics, history
def model_conv_layers(in_ls, n_layers):
    model = tf.keras.Sequential()
    if(n_layers == 1):
        model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
    if(n_layers == 2):
        model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
        model.add(tf.keras.layers.Conv1D(256, 2, activation='relu'))
    if(n_layers == 3):
        model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
        model.add(tf.keras.layers.Conv1D(256, 2, activation='relu'))
        model.add(tf.keras.layers.Conv1D(128, 2, activation='relu', input_shape=in_ls))
    if(n_layers == 4):
        model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
        model.add(tf.keras.layers.Conv1D(256, 2, activation='relu'))
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=in_ls))
        model.add(tf.keras.layers.Conv1D(64, 2, activation='relu'))
        model.add(tf.keras.layers.GlobalMaxPool1D())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3)) #Layer added to avoid the overfitting
        model.add(tf.keras.layers.Dense(1))
    return model
def model_dense_layers(in_ls, n_layers):
    data_in =  tf.keras.layers.Input(shape = in_ls, name='data_in')
    dense1 = tf.keras.layers.Dense(units = 512, activation=tf.nn.relu)
    dense2 = tf.keras.layers.Dense(units = 256, activation=tf.nn.relu)
    dense3 = tf.keras.layers.Dense(units = 128, activation=tf.nn.relu)
    dense4 = tf.keras.layers.Dense(units = 64, activation=tf.nn.relu) 
    output = tf.keras.layers.Dense(units = 1, activation=tf.nn.sigmoid)
    dropout = tf.keras.layers.Dropout(0.5)
    flattened = tf.keras.layers.Flatten()

    if(n_layers == 1):
        input = dense1(data_in)
        x = dropout(input)
        x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
        x = output(x)
    elif(n_layers == 2):
        input = dense1(data_in)
        x = dense2(input)
        x = dropout(x)
        x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
        x = output(x)

    elif(n_layers == 3):
        input = dense1(data_in)
        x = dense2(input)
        x = dense3(x)
        x = dropout(x)
        x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
        x = output(x)

    elif(n_layers == 4):
        input = dense1(data_in)
        x = dense2(input)
        x = dense3(x)
        x = dense4(x)
        x = dropout(x)
        x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
        x = output(x)

    return tf.keras.models.Model(inputs = data_in, outputs = x)

#OPTIMIZING FUNCTION TESTING
def test_opt_func(opt_func, IN_LS, TR_D, TR_L, TR_BATCH_SIZE:float, TE_D, TE_L):
    models = []
    var_metrics = []
    history =[]
    N_models = len(opt_func)
    for i in range(N_models):
        models.append(model_conv_layers(IN_LS, n_layers = 4))

    for i in range(N_models):
        models[i].compile(loss='mean_squared_error', optimizer = opt_func[i], metrics = [tf.keras.metrics.MeanSquaredError()])

        start_time = time.time()
        #FITTING
        models[i].fit(TR_D, TR_L, epochs=10, batch_size=TR_BATCH_SIZE, verbose=1) #I think becasuse of the size of the batch     
                                                                            # it is charging 28800 number of data per epoch.
        #TESTING
        print('*TEST*')
        metrics = models[i].evaluate(TE_D, TE_L)
        var_metrics = np.append(var_metrics, metrics[1])
        print("%s: %2.f%%" % (models[i].metrics_names[1], metrics[1]*100))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')

    print('\n OPTIMIZING FUNCTIONS TYPES DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    return models, var_metrics, history

#LOSS FUNCTION
def test_loss_func(loss_func, loss_func_metrics, IN_LS, TR_D, TR_L, TR_BATCH_SIZE:float, TE_D, TE_L):
    opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
    models = []
    var_metrics = []
    history =[]
    N_models = len(loss_func)
    for i in range(N_models):
        models.append(model_conv_layers(IN_LS, n_layers = 4))

    for i in range(N_models):
        models[i].compile(loss= loss_func[i], optimizer = opt_func[2], metrics = [loss_func_metrics[i]])
        history.append(models[i].fit(TR_D, TR_L, epochs=8, batch_size=TR_BATCH_SIZE, verbose=1)) #I think becasuse of the size of the batch     

        start_time = time.time()
        #FITTING
        models[i].fit(TR_D, TR_L, epochs=10, batch_size=TR_BATCH_SIZE, verbose=1) #I think becasuse of the size of the batch     
                                                                        # it is charging 28800 number of data per epoch.
        #TESTING
        print('*TEST*')
        metrics = models[i].evaluate(TE_D, TE_L)
        var_metrics = np.append(var_metrics, metrics[1])
        print("%s: %2.f%%" % (models[i].metrics_names[1], metrics[1]*100))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')

    print('\n LOSS FUNCTIONS DONE TYPES DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    return models, var_metrics, history

#LEARNING RATE
def test_lr(lr, IN_LS, TR_D, TR_L, TR_BATCH_SIZE:float, TE_D, TE_L):
    models = []
    var_metrics = []
    N_models = len(lr)
    loss_func = tf.keras.losses.MeanSquaredError()
    loss_func_metrics = tf.keras.metrics.MeanSquaredError()
    history =[]
    for i in range(N_models):
        models.append(model_conv_layers(IN_LS, n_layers = 4))

    for i in range(N_models):
        models[i].compile(loss= tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(learning_rate=lr[i]), metrics = [tf.keras.metrics.MeanSquaredError()])
        history.append(models[i].fit(TR_D, TR_L, epochs=8, batch_size=TR_BATCH_SIZE, verbose=1)) #I think becasuse of the size of the batch     

        start_time = time.time()
        #TESTING
        print('*TEST*')
        metrics = models[i].evaluate(TE_D, TE_L)
        var_metrics = np.append(var_metrics, metrics[1])
        print("%s: %2.f%%" % (models[i].metrics_names[1], metrics[1]*100))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')

    print('\n LEARNING RATE VARIATION DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    return models, var_metrics, history

#BATCH SIZE
def test_batch_size(batch_sizes, IN_LS, TR_D, TR_L, TR_BATCH_SIZE:float, TE_D, TE_L):
    models = []
    var_metrics = []
    N_models = len(batch_sizes)
    loss_func = tf.keras.losses.MeanSquaredError()
    loss_func_metrics = tf.keras.metrics.MeanSquaredError()
    history =[]
    for i in range(N_models):
        models.append(model_conv_layers(IN_LS, n_layers = 4))

    for i in range(N_models):
        models[i].compile(loss= tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), metrics = [tf.keras.metrics.MeanSquaredError()])
        history.append(models[i].fit(TR_D, TR_L, epochs=8, batch_size=batch_sizes[i], verbose=1)) #I think becasuse of the size of the batch     

        start_time = time.time()
        #TESTING
        print('*TEST*')
        metrics = models[i].evaluate(TE_D, TE_L)
        var_metrics = np.append(var_metrics, metrics[1])
        print("%s: %2.f%%" % (models[i].metrics_names[1], metrics[1]*100))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')
                                                          # it is charging 28800 number of data per epoch.
    print('\n BATCH SIZE VARIATION DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')