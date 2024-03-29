import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from train_generate.data_class import inverse_scaling
import os
from opt_depth_params import opt_len_val



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
    x = conv2(x)
    x = conv3(x)
    x = dropout(x)
    x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here

    return x

class NN_ModelCompileMixin():
    def compile_model(self, learning_rate=0.001):
        data_in =  tf.keras.layers.Input(shape = self.in_ls, name='data_in')
        output = tf.keras.layers.Dense(self.output_ravel_shape, activation=tf.nn.sigmoid)
        output_layer = output(hidden_layers(data_in))
        self.model = tf.keras.models.Model(inputs = data_in, outputs = output_layer)
        opt_func = tf.keras.optimizers.Adam(#learning_rate=learning_rate
        )
        self.model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = ["mean_squared_error"])
        self.model.summary()
    def check_create_dirs(self, intermediate_dir: str):
    #Checking the current path of directories is created
        check_path_dirs = os.getcwd()+"/"
        dir_path = [self.nn_model_type, intermediate_dir, self.light_type]
        for dir in dir_path:
            check_dirs = list(os.walk(check_path_dirs))
            bool_create = any(item in check_dirs[0][1] for item in [dir])
            check_path_dirs += dir+"/"
            if bool_create:
                None
            else:
                os.mkdir(f"{check_path_dirs}")
        return check_path_dirs
    def plot_loss(self):
        fig,ax = plt.subplots(figsize = (10,7))
        print(self.history.history)
        loss = np.array(self.history.history['loss'])
        epochs = range(len(loss))
        print(loss)
        ax.plot(epochs, loss)
        ax.set_yscale("log")
        ax.set_title(self.plot_title)
        ax.set_xlim((epochs[0], epochs[-1]))
        ax.set_ylabel("Loss")
        ax.set_xlabel("epochs")

        #Creation of the images directory
        dir_path = self.check_create_dirs("Images")
        loss_path = self.check_create_dirs("Loss")
        #Saving the plot figure
        fig.savefig(dir_path + f"loss_plot-{self.filename}.png")
        np.save(loss_path+"loss-"+self.filename+".npy", loss)
        print(f"{self.filename} loss plotted!")

    #Load the weights for the NN model if they are already created
    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

################################################################################################################
# ATMOSPHERE PARAMETER PREDICTING FEATURES
################################################################################################################

class AtmTrainVisualMixin():
    """
    This generates the atmosphere parameters output from an input of Stokes parameters
    """
    def __init__(self):
        self.plot_title = "Atmosphere parameters"
        self.nn_model_type = "atm_NN_model"
        self.length = opt_len_val() #atmosphere length
        self.scaler_names = ["mbyy", "mvyy", "mrho", "mtpr"]
        self.title = ['Magnetic Field','Velocity','Density','Temperature']
        self.channels = len(self.scaler_names)

        self.in_ls = (self.nlam, 4)
        self.output_ravel_shape = self.length*self.channels

    def train(self,filename, tr_s=0.75, batch_size=2, epochs=8):
        """
        tr_s: training size percentage
        """
        #Charging the splitted atm model data
        self.split_data_atm_output(filename, self.light_type, tr_s)

        #Checking the path of directories is created
        dir_path = self.check_create_dirs("training")

        #checkpoint model specifications
        checkpoint_path = dir_path+f"cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=0)
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[cp_callback])
        self.model.evaluate(self.te_input, self.te_output, verbose=0)
    def predict_values(self, filename):

        #Charging and reshaping the light values
        
        if self.light_type == "Intensity":
            self.charge_intensity(filename)
            print(f"{self.filename} predicting...")
            predicted_values = self.model.predict(self.iout.flatten(order = "C"))
        if self.light_type == "Stokes params":
            self.charge_stokes_params(filename)
            print(f"{self.filename} predicting...")
            predicted_values = self.model.predict(np.memmap.reshape(self.profs, (self.nx*self.nz, self.nlam, 4)))
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.channels, self.length))
        predicted_values = np.moveaxis(predicted_values,2,3)


        #Inverse scaling application
        for i in range(self.channels):
            predicted_values[:,:,:,i] = np.memmap.reshape(inverse_scaling(predicted_values[:,:,:,i], self.scaler_names[i]), (self.nx,self.nz,self.length))
        print(f"{filename} prediction done!")

        check_path_dirs = os.getcwd()+"/"
        dir_path = [self.nn_model_type, "Predicted_values", self.light_type]

        #Checking the current path of directories is created
        dir_path = self.check_create_dirs("Predicted_values")
        
        #Saving the predicted values
        np.save(dir_path+f"obtained_value-{filename}.npy", predicted_values)
        return predicted_values
    def plot_predict(self, plot_file):
        ix = 200
        iz = 280
        height = 10
        fig, ax = plt.subplots(4,4,figsize=(50,7))
        predicted_values = np.load(f"{self.nn_model_type}/Predicted_values/{self.light_type}/obtained_value-{plot_file}.npy")
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.length,self.channels))
        print(np.argwhere(np.isnan(predicted_values)))
        original_atm = self.remmap_opt_depth(plot_file)
        original_atm = np.memmap.reshape(original_atm, (self.nx, self.nz, self.length,self.channels))
        print("original data array shape:",original_atm.shape)
        for i in range(self.channels):
            original_atm[:,:,:,i] = np.memmap.reshape(inverse_scaling(original_atm[:,:,:,i], self.scaler_names[i]), (self.nx,self.nz,self.length))
        
        #Checking the path of directories is created
        dir_path = self.check_create_dirs("Images")

        #Loading and plotting the predicted values vs the original ones

        for i in range(self.channels):
            ax[0,i].plot(range(self.length), predicted_values[ix,iz,:,i], label="Predicted curve")
            ax[0,i].set_title(f"Atmosphere parameters height serie - title={self.title[i]} - ix={ix}, iy={iz}")
            ax[0,i].plot(range(self.length), original_atm[ix,iz,:,i], label="Original curve")
            ax[0,i].legend()
            ax[1,i].imshow(predicted_values[:,:,height,i], cmap = "gist_gray")     
            ax[1,i].set_title(f"Atmosphere parameters spatial distribution- title={self.title[i]}")
            ax[2,i].imshow(original_atm[:,:,height,i], cmap = "gist_gray")     
            ax[2,i].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")
            ax[3,i].imshow(np.abs(np.subtract(original_atm[:,:,height,i], predicted_values[:,:,height,i])), cmap = "gist_gray")     
            ax[3,i].set_title(f"Substraction of both images - title={self.title[i]}")

            fig.savefig(dir_path + f"Atmosphere_parameter-{plot_file}.png")
        print(f"{plot_file} prediction plotted\n")

################################################################################################################
# LIGHT PREDICTING FEATURES
################################################################################################################


class LightTrainVisualMixin():
    """
    This generates the Stokes parameters output from an input of Atm parameters
    """
    def __init__(self):

        self.nn_model_type = "light_NN_model"
        self.length = self.nlam
        if self.light_type == "Intensity":
            self.scaler_name = "iout"
            self.plot_title = "Intensity"
        if self.light_type == "Stokes params":
            self.scaler_name = "stokes"
            self.plot_title = "Stokes parameters"
        self.title = ['I stokes','Q stokes','U stokes','V stokes']
        self.channels = 4

        self.in_ls = (opt_len_val(), 4)
        self.output_ravel_shape = self.length*self.channels
        
    def train(self,filename, tr_s=0.75, batch_size=2, epochs=8):
        """
        tr_s: training size percentage
        """
        #Charging the splitted light model data
        self.split_data_light_output(filename, self.light_type, tr_s)

        #Checking the path of directories is created
        dir_path = self.check_create_dirs("training")
        
        #checkpoint model specifications
        checkpoint_path = dir_path+f"cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=0)
        self.batch_size = batch_size
        self.epochs = epochs
        print("output shape:", self.tr_output.shape)
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[cp_callback])
        self.model.evaluate(self.te_input, self.te_output)
    def predict_values(self, filename):
        self.filename = filename
        self.remmap_opt_depth(filename)
        predicted_values = self.model.predict(np.memmap.reshape(self.atm_params, (self.nx*self.nz, self.length, 4)))
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.channels, self.length))
        #Inverse scaling application
        if self.light_type == "Intensity":
            predicted_values = np.memmap.reshape(inverse_scaling(predicted_values, self.scaler_name), (self.nx,self.nz))
        if self.light_type == "Stokes params":
            predicted_values = np.memmap.reshape(inverse_scaling(predicted_values, self.scaler_name), (self.nx,self.nz,4,self.length))
        print(f"{filename} prediction done!")

        #Checking the path of directories is created
        dir_path = self.check_create_dirs("Predicted Values")

        #Saving the predicted values
        np.save(dir_path+f"obtained_value-{filename}.npy", predicted_values)
        return predicted_values
    def plot_predict(self, plot_file):
        ix = 200
        iz = 280
        lam = 10
        

        #Loading and plotting the predicted values vs the original ones
        dir_path = self.check_create_dirs("Predicted_values")
        predicted_values = np.load(dir_path + f"obtained_value-{plot_file}.npy")
        
        #Checking the path of directories is created
        dir_path = self.check_create_dirs("Images")

        #Intensity
        if self.light_type == "Intensity":
            original_iout = self.charge_intensity(plot_file)
            original_iout = np.memmap.reshape(inverse_scaling(original_iout, self.scaler_name), (self.nx,self.nz))
            print(f"{plot_file} prediction done!")
            fig, ax = plt.subplots(1,4,figsize=(50,7))
            ax[0].imshow(predicted_values, cmap = "gist_gray")     
            ax[0].set_title(f"Intensity spatial distribution- title={self.title[i]}")
            ax[1].imshow(original_iout, cmap = "gist_gray")     
            ax[1].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")
            ax[2].imshow(np.abs(np.subtract(original_iout, predicted_values)), cmap = "gist_gray")     
            ax[2].set_title(f"Substraction of both images - title={self.title[i]}")
        
        #Stokes params
        if self.light_type == "Stokes parameters":
            original_stokes = self.charge_stokes_params(plot_file)
            original_stokes = np.memmap.reshape(inverse_scaling(original_stokes, self.scaler_name), (self.nx,self.nz,self.length))
            fig, ax = plt.subplots(4,4,figsize=(50,7))


            for i in range(self.channels):
                ax[0,i].plot(range(self.length), predicted_values[ix,iz,:,i], label="Predicted curve")
                ax[0,i].set_title(f"Stokes parameters wavelength serie - title={self.title[i]} - ix={ix}, iy={iz}")
                ax[0,i].plot(range(self.length), original_stokes[ix,iz,:,i], label="Original curve")
                ax[0,i].legend()
                ax[1,i].imshow(predicted_values[:,:,lam,i], cmap = "gist_gray")     
                ax[1,i].set_title(f"Stokes parameters spatial distribution- title={self.title[i]}")
                ax[2,i].imshow(original_stokes[:,:,lam,i], cmap = "gist_gray")     
                ax[2,i].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")
                ax[3,i].imshow(np.abs(np.subtract(original_stokes[:,:,lam,i], predicted_values[:,:,lam,i])), cmap = "gist_gray")     
                ax[3,i].set_title(f"Substraction of both images - title={self.title[i]}")

            fig.savefig(dir_path + f"Atmosphere_parameter-{plot_file}.png")
        print(f"{plot_file} prediction plotted\n")


