import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_class import Data_class
import testing_functions as tef

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    for i in range(53,60):
        a = "0"+str(i)+"000"
        tr_filename.append(a)
    IN_LS = np.array([4,256]) #input shape in input layer

    sun_model = NN_model()
    sun_model.compile_model(IN_LS)
    for fln in tr_filename:
        sun_model.train(fln, "Intensity", tr_s = 0.75, batch_size_percentage = 0.05, epochs=3)
        sun_model.plot_loss()

class NN_model(Data_class):
    def __init__(self, nx = 480, ny = 256, nz = 480):
        super().__init__(nx,ny,nz)
    def compile_model(self, in_ls):
        data_in =  tf.keras.layers.Input(shape = in_ls, name='data_in')
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
        opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        self.model.summary()
        return self.model
    def train(self,filename, output_type, tr_s, batch_size_percentage, epochs=8):
        """
        batch_size: its a fraction relative to the total of the set (must be between 0<x<1).
        """
        self.split_data(filename, output_type, tr_s)
        TR_BATCH_SIZE = int(self.tr_input[:,1,2].size*batch_size_percentage)
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=epochs, batch_size=TR_BATCH_SIZE, verbose=1)
        self.model.evaluate(self.te_input, self.te_output)
    def plot_loss(self):
        fig,ax = plt.subplots(figsize = (10,7))
        ax.plot(range(self.history['loss'].size), self.history['loss'])
        fig.savefig(f"Images/loss_plot-{self.filename}.png")
        print(f"{self.filename} loss plotted!")

if __name__ == "__main__":
    main()