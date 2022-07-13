import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_class import Data_class
import testing_functions as tef

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = "053000"

    data = Data_class()
    TR_D, TR_L, TE_D, TE_L = data.split_data(tr_filename, TR_S = 0.75, output_type="Intensity")

    IN_LS = np.array([4,256]) #input shape in input layer
    TR_BATCH_SIZE = int(len(TR_D[:,1,2])/1)

    #model = tef.model_dense_layers(IN_LS, n_layers = 4)
    #opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
    #model.summary()
    #model.fit(TR_D, TR_L, epochs=8, batch_size=TR_BATCH_SIZE, verbose=1)
    model = NN_model(IN_LS, TR_D, TR_L, TE_D, TE_L, TR_BATCH_SIZE)
    model.model_train()

class NN_model():
    def __init__(self, IN_LS, TR_D, TR_L, TE_D, TE_L, TR_BATCH_SIZE):
        self.tr_input = TR_D
        self.tr_output = TR_L
        self.te_input = TE_D
        self.te_output = TE_L
        self.in_ls = IN_LS
        self.tr_batch_size = TR_BATCH_SIZE
    def model_train(self):
        self.model = compile_model(self.in_ls)
        opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        self.model.summary()
        self.model.fit(self.tr_input, self.tr_output, epochs=8, batch_size=self.tr_batch_size, verbose=1)

def compile_model(IN_LS):
    data_in =  tf.keras.layers.Input(shape = IN_LS, name='data_in')
    dense1 = tf.keras.layers.Conv1D(units = 512, activation=tf.nn.relu)
    dense2 = tf.keras.layers.Conv1D(units = 256, activation=tf.nn.relu)
    dense3 = tf.keras.layers.Conv1D(units = 128, activation=tf.nn.relu)
    dense4 = tf.keras.layers.Conv1D(units = 64, activation=tf.nn.relu) 
    output = tf.keras.layers.Conv1D(units = 1, activation=tf.nn.sigmoid)
    dropout = tf.keras.layers.Dropout(0.5)
    flattened = tf.keras.layers.Flatten()
    
    input = dense1(data_in)
    x = dense2(input)
    x = dense3(x)
    x = dense4(x)
    x = dropout(x)
    x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
    x = output(x)

    return tf.keras.models.Model(inputs = data_in, outputs = x)




if __name__ == "__main__":
    main()