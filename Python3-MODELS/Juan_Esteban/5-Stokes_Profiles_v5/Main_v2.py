import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_class import Data_class
import testing_functions as tef

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = "053000"
    IN_LS = np.array([4,256]) #input shape in input layer

    sun_model = NN_model()
    sun_model.train(tr_filename, "Intensity", IN_LS, TR_S = 0.75, batch_size = 0.05)

class NN_model(Data_class):
    def __init__(self, nx = 480, ny = 256, nz = 480):
        super().__init__(nx,ny,nz)
    def compile_model(self, IN_LS):
        self.in_ls
        data_in =  tf.keras.layers.Input(shape = self.in_ls, name='data_in')
        dense1 = tf.keras.layers.Conv1D(512, 2, activation=tf.nn.relu)
        dense2 = tf.keras.layers.Conv1D(256, 2, activation=tf.nn.relu)
        dense3 = tf.keras.layers.Conv1D(128, 2, activation=tf.nn.relu)
        dense4 = tf.keras.layers.Conv1D(64, 1, activation=tf.nn.relu) 
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        dropout = tf.keras.layers.Dropout(0.5)
        flattened = tf.keras.layers.Flatten()
        
        input = dense1(data_in)
        x = dense2(input)
        x = dense3(x)
        x = dense4(x)
        x = dropout(x)
        x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
        x = output(x)

        self.model = tf.keras.models.Model(inputs = data_in, outputs = x)
        return self.model
    def train(self,filename, output_type, TR_S, IN_LS, batch_size):
        """
        batch_size: its a fraction relative to the total of the set (must be between 0<x<1).
        """
        self.split_data(filename, output_type, TR_S)
        self.compile_model(IN_LS)
        opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        self.model.summary()
        TR_BATCH_SIZE = int(self.tr_input[:,1,2].size*batch_size)
        self.model.fit(self.tr_input, self.tr_output, epochs=8, batch_size=TR_BATCH_SIZE, verbose=1)

if __name__ == "__main__":
    main()