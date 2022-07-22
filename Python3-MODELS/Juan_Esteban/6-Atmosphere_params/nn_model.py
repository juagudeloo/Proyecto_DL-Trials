import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_class import Data_class

class NN_model_atm(Data_class):
    def __init__(self, input_type, nx = 480, ny = 256, nz = 480):
        super().__init__(nx,ny,nz)
        self.input_type = input_type
    def compile_model(self, in_ls):
        data_in =  tf.keras.layers.Input(shape = in_ls, name='data_in')
        conv1 = tf.keras.layers.Conv1D(512, 2, activation=tf.nn.relu)
        conv2 = tf.keras.layers.Conv1D(256, 2, activation=tf.nn.relu)
        conv3 = tf.keras.layers.Conv1D(128, 2, activation=tf.nn.relu)
        conv4 = tf.keras.layers.Conv1D(64, 1, activation=tf.nn.relu) 
        dropout = tf.keras.layers.Dropout(0.5)
        flattened = tf.keras.layers.Flatten()
        if self.input_type == "Intensity":
            output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        if self.input_type == "Stokes params":   
            output = tf.keras.layers.Dense(4*256, activation=tf.nn.sigmoid)

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
        self.split_data(filename, self.input_type, tr_s)
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.evaluate(self.te_input, self.te_output)
    def plot_loss(self):
        fig,ax = plt.subplots(figsize = (10,7))
        ax.plot(range(len(self.history.history['loss'])), self.history.history['loss'])
        ax.set_title("Atmosphere parameters")
        ax.set_ylabel("Loss")
        ax.set_xlabel("epochs")
        if self.input_type == "Intensity":
            fig.savefig(f"Images/Intensity/loss_plot-{self.filename}.png")
        if self.input_type == "Stokes params":
            fig.savefig(f"Images/loss_plot-{self.filename}.png")
        print(f"{self.filename} loss plotted!")
    ##### PREDICTING PHASE #####
    def predict_values(self, filename):
        self.pred_filename = filename
        print(f"{self.pred_filename} predicting...")
        if self.input_type == "Intensity":
            self.charge_intensity(self.pred_filename)
            self.predicted_values = np.memmap.reshape(self.model.predict(self.iout), (self.nx, self.nz, 4, self.ny))
            print(f"{self.pred_filename} prediction done!")
        if self.input_type == "Stokes params":
            self.charge_stokes_params(self.pred_filename)
            self.predicted_values = np.memmap.reshape(self.model.predict(self.profs), (self.nx, self.nz, 4, self.ny))
            print(f"{self.pred_filename} prediction done!\n")
        return self.predicted_values
    def plot_predict(self):
        N_profs = 4
        ix = 200
        iz = 280
        height = 200
        title = ['Magnetic Field','Velocity','Density','Temperature']
        fig, ax = plt.subplots(3,4,figsize=(30,7))
        original_atm = self.charge_atm_params(self.pred_filename)
        original_atm = np.memmap.reshape(self.nx, self.nz, 4, self.ny)
        for i in range(N_profs):
            ax[0,i].plot(range(self.ny), self.predicted_values[ix,iz,i,:])
            ax[0,i].set_title(f"Atmosphere parameters height serie - title={title[i]} - ix={ix}, iy={iz}")
            ax[1,i].plot(range(self.ny), original_atm[ix,iz,i,:])
            ax[1,i].set_title(f"ORIGINAL height serie - title={title[i]} - ix={ix}, iy={iz}")
            ax[2,i].imshow(self.predicted_values[:,:,i,height], cmap = "gist_gray")     
            ax[2,i].set_title(f"Atmosphere parameters spatial distribution- title={title[i]}")
            ax[3,i].imshow(original_atm[:,:,i,height], cmap = "gist_gray")     
            ax[3,i].set_title(f"ORIGINAL spatial distribution - title={title[i]}")
        if self.input_type == "Intensity":
            fig.savefig(f"Images/Intensity/loss_plot-{self.filename}.png")
        if self.input_type == "Stokes params":
            fig.savefig(f"Images/Stokes_params/loss_plot-{self.filename}.png")
        print(f"{self.pred_filename} prediction plotted\n")
