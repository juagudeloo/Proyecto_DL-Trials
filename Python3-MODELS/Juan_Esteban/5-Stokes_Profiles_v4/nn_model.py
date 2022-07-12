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
        self.in_ls = IN_LS
        self.out_ls = OUT_LS
    def compile_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
        self.model.add(tf.keras.layers.Conv1D(256, 2, activation='relu'))
        self.model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=self.in_ls))
        self.model.add(tf.keras.layers.Conv1D(64, 2, activation='relu'))
        self.model.add(tf.keras.layers.GlobalMaxPool1D())
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.3)) #Layer added to avoid the overfitting
        self.model.add(tf.keras.layers.Dense(self.out_ls))
        lr = 0.001
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = keras.metrics.MeanSquaredError()
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



