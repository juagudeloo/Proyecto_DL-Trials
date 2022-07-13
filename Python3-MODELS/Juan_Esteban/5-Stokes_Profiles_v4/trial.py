import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D, Input, GlobalMaxPool1D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from data_class import Data_NN_model

def main():
    print("compiling the model...")
    model = tf.keras.Sequential()
    model.add(Conv1D(512, 2, activation='relu', input_shape=self.in_ls))
    model.add(Conv1D(256, 2, activation='relu'))
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3)) #Layer added to avoid the overfitting
    model.add(Dense(self.out_ls, name="output"))
    lr = 0.001
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.metrics.MeanSquaredError()
    model.compile(optimizer = opt, loss = loss, metrics = loss)
    print("model compiled!")
    print(mode.summary())

    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = "053000"
    IN_LS = (4,256) #input shape in input layer
    data_obj = Data_NN_model(IN_LS, 1, "Intensity")

if __name__ == "__main__":
    main()