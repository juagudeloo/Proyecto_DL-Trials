import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_class import inverse_scaling
import os

################################################################################################################
# NN FEATURES
################################################################################################################

def hidden_layers(input_layer):
    conv1 = tf.keras.layers.Conv1D(512, kernel_size = 2, activation=tf.nn.relu)
    conv2 = tf.keras.layers.Conv1D(256, kernel_size = 2, activation=tf.nn.relu)
    conv3 = tf.keras.layers.Conv1D(64, kernel_size = 2, activation=tf.nn.relu) 
    dropout = tf.keras.layers.Dropout(0.5)
    flattened = tf.keras.layers.Flatten()

    x = conv1(input_layer)
    x = conv2(input)
    x = conv3(x)
    x = dropout(x)
    x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here

    return x

def set_model_params_and_compilation(model, input_layer, x_layer, learning_rate):
    model = tf.keras.models.Model(inputs = input_layer, outputs = x_layer)
    opt_func = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
    model.summary()

class NN_ModelCompileMixin():
    def compile_model(self, learning_rate=0.001):
        data_in =  tf.keras.layers.Input(shape = self.in_ls, name='data_in')
        output = tf.keras.layers.Dense(self.output_ravel_shape, activation=tf.nn.sigmoid)
        x = output(hidden_layers(data_in))
        set_model_params_and_compilation(self.model, data_in, x, learning_rate)
        
    def plot_loss(self):
        fig,ax = plt.subplots(figsize = (10,7))
        ax.plot(range(len(self.history.history['loss'])), self.history.history['loss'])
        ax.set_title(self.plot_title)
        ax.set_ylabel("Loss")
        ax.set_xlabel("epochs")
        fig.savefig(f"{self.nn_model_type}/Images/{self.light_type}/loss_plot-{self.filename}.png")
        print(f"{self.filename} loss plotted!")
    ##### PREDICTING PHASE #####
    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

################################################################################################################
# ATMOSPHERE PARAMETER PREDICTING FEATURES
################################################################################################################

class AtmTrainVisualMixin():
    def __init__(self, light_type = "Intensity"):
        self.plot_title = "Atmosphere parameters"
        self.nn_model_type = "atm_NN_model"
        self.light_type = light_type
        self.length = 256-self.lb
        self.scaler_names = ["mbyy", "mvyy", "mrho", "mtpr"]
        self.title = ['Magnetic Field','Velocity','Density','Temperature']
        self.channels = len(self.scaler_names)
        self.output_ravel_shape = self.length*self.channels
    def train(self,filename,tr_s=0.75, batch_size=2, epochs=8):
        """
        tr_s: training size percentage
        """
        self.split_data_atm_output(filename, self.light_type, tr_s)

        #checkpoint model specifications
        checkpoint_path = f"{self.nn_model_type}/training/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=[cp_callback])
        self.model.evaluate(self.te_input, self.te_output)
    def predict_values(self, filename):
        print(f"{self.filename} predicting...")
        if self.light_type == "Intensity":
            self.charge_intensity(filename)
            predicted_values = self.model.predict(self.iout)
        if self.light_type == "Stokes params":
            self.charge_stokes_params(filename)
            predicted_values = self.model.predict(self.profs)
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.channels, self.length))
        #Inverse scaling application
        for i in range(self.channels):
            predicted_values[:,:,i,:] = np.memmap.reshape(inverse_scaling(predicted_values[:,:,i,:], self.scaler_names[i]), (self.nx,self.nz,self.length))
        print(f"{filename} prediction done!")
        np.save(f"{self.nn_model_type}/Predicted_values/obtained_value-{filename}.npy", predicted_values)
        return predicted_values
    def plot_predict(self, filename):
        ix = 200
        iz = 280
        height = 10
        fig, ax = plt.subplots(4,4,figsize=(50,7))
        predicted_values = np.load(f"{self.nn_model_type}/Predicted_values/obtained_value-{filename}.npy", predicted_values)
        original_atm = self.charge_atm_params(filename)
        original_atm = np.memmap.reshape(original_atm, (self.nx, self.nz, self.channels, self.length))
        for i in range(self.channels):
            original_atm[:,:,i,:] = np.memmap.reshape(inverse_scaling(original_atm[:,:,i,:], self.scaler_names[i]), (self.nx,self.nz,self.length))
        print(f"{filename} prediction done!")
        for i in range(self.channels):
            ax[0,i].plot(range(self.length), predicted_values[ix,iz,i,:], label="Predicted curve")
            ax[0,i].set_title(f"Atmosphere parameters height serie - title={self.title[i]} - ix={ix}, iy={iz}")
            ax[0,i].plot(range(self.length), original_atm[ix,iz,i,:], label="Original curve")
            ax[0,i].legend()
            ax[1,i].imshow(predicted_values[:,:,i,height], cmap = "gist_gray")     
            ax[1,i].set_title(f"Atmosphere parameters spatial distribution- title={self.title[i]}")
            ax[2,i].imshow(original_atm[:,:,i,height], cmap = "gist_gray")     
            ax[2,i].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")
            ax[3,i].imshow(np.abs(np.subtract(original_atm[:,:,i,height], predicted_values[:,:,i,height])), cmap = "gist_gray")     
            ax[3,i].set_title(f"Substraction of both images - title={self.title[i]}")

            fig.savefig(f"{self.nn_model_type}/Images/{self.light_type}/Atmosphere_parameter-{self.filename}.png")
        print(f"{filename} prediction plotted\n")

################################################################################################################
# LIGHT PREDICTING FEATURES
################################################################################################################


class LightTrainVisualMixin():
    def __init__(self, light_type = "Intensity"):
        self.plot_title = "Stokes parameters"
        self.nn_model_type = "stokes_NN_model"
        self.light_type = light_type
        self.length = self.nlam
        if self.light_type == "Intensity":
            self.scaler_name = "iout"
        if self.light_type == "Stokes params":
            self.scaler_name = "stokes"
        self.title = ['I stokes','Q stokes','U stokes','V stokes']
        self.channels = 4
        self.output_ravel_shape = self.length*self.channels
    def train(self,filename, tr_s=0.75, batchs_size=2, epochs=8):
        """
        tr_s: training size percentage
        """
        self.split_data_light_output(filename, self.light_type, tr_s)

        #checkpoint model specifications
        checkpoint_path = f"{self.nn_model_type}/training/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=[cp_callback])
        self.model.evaluate(self.te_input, self.te_output)
    def predict_values(self, filename):
        self.charge_atm_params(filename)
        predicted_values = self.model.predict(self.atm_params)
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.channels, self.length))
        #Inverse scaling application
        if self.light_type == "Intensity":
            predicted_values = np.memmap.reshape(inverse_scaling(predicted_values, self.scaler_name), (self.nx,self.nz))
        if self.light_type == "Stokes params":
            predicted_values = np.memmap.reshape(inverse_scaling(predicted_values, self.scaler_name), (self.nx,self.nz,self.length))
        print(f"{filename} prediction done!")
        np.save(f"{self.nn_model_type}/Predicted_values/obtained_value-{filename}.npy", predicted_values)
        return predicted_values
    def plot_predict(self, filename):
        ix = 200
        iz = 280
        lam = 10
        predicted_values = np.load(f"{self.nn_model_type}/Predicted_values/obtained_value-{filename}.npy", predicted_values)
        if self.light_type == "Intensity":
            original_iout = self.charge_intensity(filename)
            original_iout = np.memmap.reshape(inverse_scaling(original_iout, self.scaler_name), (self.nx,self.nz))
            print(f"{filename} prediction done!")
            fig, ax = plt.subplots(1,4,figsize=(50,7))
            ax[0].imshow(predicted_values, cmap = "gist_gray")     
            ax[0].set_title(f"Intensity spatial distribution- title={self.title[i]}")
            ax[1].imshow(original_iout, cmap = "gist_gray")     
            ax[1].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")
            ax[2].imshow(np.abs(np.subtract(original_iout, predicted_values)), cmap = "gist_gray")     
            ax[2].set_title(f"Substraction of both images - title={self.title[i]}")

        if self.light_type == "Stokes parameters":
            original_stokes = self.charge_stokes_params(filename)
            original_stokes = np.memmap.reshape(inverse_scaling(original_stokes, self.scaler_name), (self.nx,self.nz,self.length))
            print(f"{filename} prediction done!")
            fig, ax = plt.subplots(4,4,figsize=(50,7))
            for i in range(self.channels):
                ax[0,i].plot(range(self.length), predicted_values[ix,iz,i,:], label="Predicted curve")
                ax[0,i].set_title(f"Stokes parameters wavelength serie - title={self.title[i]} - ix={ix}, iy={iz}")
                ax[0,i].plot(range(self.length), original_stokes[ix,iz,i,:], label="Original curve")
                ax[0,i].legend()
                ax[1,i].imshow(predicted_values[:,:,i,lam], cmap = "gist_gray")     
                ax[1,i].set_title(f"Stokes parameters spatial distribution- title={self.title[i]}")
                ax[2,i].imshow(original_stokes[:,:,i,lam], cmap = "gist_gray")     
                ax[2,i].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")
                ax[3,i].imshow(np.abs(np.subtract(original_stokes[:,:,i,lam], predicted_values[:,:,i,lam])), cmap = "gist_gray")     
                ax[3,i].set_title(f"Substraction of both images - title={self.title[i]}")

            fig.savefig(f"{self.nn_model_type}/Images/{self.light_type}/Atmosphere_parameter-{self.filename}.png")
        print(f"{filename} prediction plotted\n")


