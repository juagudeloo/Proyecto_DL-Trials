from pickletools import optimize
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D, Input, GlobalMaxPool1D, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import keras 
from sklearn.metrics import r2_score

class NN_Model():
    def __init__(self, IN_LS, OUT_LS):
        inputs = Input(shape = IN_LS, name = "data in")
        conv = Conv1D(512, 2, activation = 'sigmoid')(inputs)
        conv = Conv1D(256, 2, activation = 'sigmoid')(conv)
        conv = Conv1D(128, 2, activation = 'sigmoid')(conv)
        conv = Conv1D(64, 1, activation = 'sigmoid')(conv)
        max_pool = GlobalMaxPool1D()(conv)
        dense = Dense(64, activation = 'sigmoid')(max_pool)
        outputs = Dense(OUT_LS, activation = 'softmax')(dense)
        self.model = Model(inputs = inputs, outputs = outputs, name = 'project_dl', validation = 0.2)
        lr = 0.001
        opt = keras.optimizers.Adam(learning_rate = lr)
        loss = keras.metrics.MeanSquaredError(learning_rate = lr)
        self.model.compile(optimizer = opt, loss = loss, metrics = loss)
    
    

#    def predict_intensity_eval(self, data_PR, labels_PR, nx, nz, plot_intensity = False):
#        self.intensity = self.model.predict(data_PR).reshape(nx, nz)
#        self.std_orig = np.std(self.intensity)
#        self.std_pred = np.std(labels_PR)
#        self.R2 = r2_score(self.intensity.flatten, labels_PR)
#        print(f"std_orig = {self.std_orig}, std_pred = {self.std_pred}, R2 = {self.R2}")
#        if plot_intensity == True:
#            fig, ax = plt.subplots(1,2,figsize = (20,10))
#            ax[0].imshow(self.intensity)
#            ax[1].imshow(labels_PR)
#            fig.suptitle(f"std_orig = {self.std_orig}, std_pred = {self.std_pred}, R2 = {self.R2}")



