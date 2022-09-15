import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_class import Data_class

class NN_model(Data_class):
    def __init__(self, output_type, nx = 480, ny = 256, nz = 480):
        super().__init__(nx,ny,nz)
        self.output_type = output_type
    def compile_model(self, in_ls):
        data_in =  tf.keras.layers.Input(shape = in_ls, name='data_in')
        conv1 = tf.keras.layers.Conv1D(512, 2, activation=tf.nn.relu)
        conv2 = tf.keras.layers.Conv1D(256, 2, activation=tf.nn.relu)
        conv3 = tf.keras.layers.Conv1D(128, 2, activation=tf.nn.relu)
        conv4 = tf.keras.layers.Conv1D(64, 1, activation=tf.nn.relu) 
        dropout = tf.keras.layers.Dropout(0.5)
        flattened = tf.keras.layers.Flatten()
        if self.output_type == "Intensity":
            output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        if self.output_type == "Stokes params":   
            output = tf.keras.layers.Dense(1200, activation=tf.nn.sigmoid)

        input = conv1(data_in)
        x = conv2(input)
        x = conv3(x)
        x = conv4(x)
        x = dropout(x)
        x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
        x = output(x)

        self.model = tf.keras.models.Model(inputs = data_in, outputs = x)
        opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        self.model.summary()
        
        return self.model
    def train(self,filename, tr_s, batch_size, epochs=8):
        """
        batch_size: its a fraction relative to the total of the set (must be between 0<x<1).
        """
        self.split_data(filename, self.output_type, tr_s)
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.evaluate(self.te_input, self.te_output)
    def plot_loss(self):
        fig,ax = plt.subplots(figsize = (10,7))
        ax.plot(range(len(self.history.history['loss'])), self.history.history['loss'])
        if self.output_type == "Intensity":
            fig.savefig(f"Images/Intensity/loss_plot-{self.filename}.png")
        if self.output_type == "Stokes params":
            fig.savefig(f"Images/Stokes_params/loss_plot-{self.filename}.png")
        print(f"{self.filename} loss plotted!")
    ##### PREDICTING PHASE #####
    def predict_values(self, filename):
        self.charge_inputs(filename)
        print(f"{self.filename} predicting...")
        if self.output_type == "Intensity":
            self.predicted_values = self.model.predict(self.input_values).reshape(self.nx, self.nz)
            print(f"{self.filename} prediction done!")
        if self.output_type == "Stokes params":
            self.predicted_values = self.model.predict(self.input_values).reshape(self.nx, self.nz, 4, self.nlam)
            print(f"{self.filename} prediction done!\n")
        return self.predicted_values
    def plot_predict(self):
        if self.output_type == "Intensity":
            fig, ax = plt.subplots(figsize = (7,7))
            ax.imshow(self.predicted_values)
            ax.set_title(f"Predicted intensity")
            fig.savefig(f"Images/Intensity/Predicted_intensity-{self.filename}.png")
        if self.output_type == "Stokes params":
            N_profs = 4
            ix = 200
            iz = 280
            wave_lam = 200
            title = ['I','Q','U','V']
            ylabel = [r'$I$ [ph]',r'$Q$ [ph]',r'$U$ [ph]',r'$V$ [ph]']
            fig, ax = plt.subplots(2,4,figsize=(30,7))
            for i in range(N_profs):
                ax[0,i].plot(np.arange(6302,6302+10*self.nlam, 10), self.predicted_values[ix,iz,i,:])
                ax[0,i].set_title(f"Stokes params spectra - ix={ix}, iy={iz}")
                ax[0,i].set_xlabel(r"$\lambda$ [$\AA$]")
                ax[0,i].set_ylabel(ylabel[i])
                ax[1,i].imshow(self.predicted_values[:,:,i,wave_lam], cmap = "gist_gray")
                ax[1,i].scatter(ix, iz, "r", label = "Spectra point")                     
                ax[1,i].set_title(f"Stokes params spatial distribution- title={title[i]}")
            fig.savefig(f"Images/Stokes_params/Predicted_Stokes_parameters-{self.filename}.png")   
        print(f"{self.filename} prediction plotted\n")
