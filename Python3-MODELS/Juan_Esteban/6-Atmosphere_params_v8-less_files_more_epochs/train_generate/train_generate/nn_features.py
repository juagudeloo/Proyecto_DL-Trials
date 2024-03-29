import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from train_generate.data_class import inverse_scaling
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
        loss = self.history.history['loss']
        epochs = range(len(loss))
        print(loss)
        ax.plot(epochs, loss)
        ax.set_title(self.plot_title)
        ax.set_xlim((epochs[0], epochs[-1]))
        ax.set_ylabel("Loss")
        ax.set_xlabel("epochs")

        #Creation of the images directory
        dir_path = self.check_create_dirs("Images")

        #Saving the plot figure
        fig.savefig(dir_path + f"loss_plot-{self.filename}.png")
        print(f"{self.filename} loss plotted!")

    #Load the weights for the NN model if they are already created
    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

################################################################################################################
# ATMOSPHERE PARAMETER PREDICTING FEATURES
################################################################################################################

class AtmTrainVisualMixin():
    def __init__(self):
        self.plot_title = "Atmosphere parameters"
        self.nn_model_type = "atm_NN_model"
        self.length = 256-self.lb
        if self.light_type == "Intensity":
            self.light_scaler_name = "iout"
        if self.light_type == "Stokes params":
            self.light_scaler_name = "stokes"
        self.atm_scaler_names = ["mbyy", "mvyy", "mrho", "mtpr"]

        #the density is the logarithm base 10 of the actual density
        self.atm_title = [r'$B$ [G]',r'$v$ [m s$^{-1}$]',r'$log(rho)$',r'$T$ [K]']
        self.light_title = ['I stokes','Q stokes','U stokes','V stokes']

        self.channels = len(self.atm_scaler_names)
        self.output_ravel_shape = self.length*self.channels
        self.in_ls = (300, 4)
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
                                                        verbose=1)
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[cp_callback])
        self.model.evaluate(self.te_input, self.te_output)
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
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.length, self.channels))

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
    def plot_predict_initial(self, filename, ix = 200, iz = 280, height = 10, ilam = 20): 
        """
        This function makes an initial plot of the Stokes parameters to check where to to choose granular or intergranular zones where
        to plot the magnitudes along.
        """
        self.filename = filename
        fig, ax = plt.subplots(1,4,figsize=(50,7))
        predicted_values = np.load(f"{self.nn_model_type}/Predicted_values/{self.light_type}/obtained_value-{self.filename}.npy")
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.length,self.channels))
        original_atm = self.charge_atm_params(self.filename)
        original_atm = np.memmap.reshape(original_atm, (self.nx, self.nz, self.length,self.channels))
        for i in range(self.channels):
            original_atm[:,:,:,i] = np.memmap.reshape(inverse_scaling(original_atm[:,:,:,i], self.atm_scaler_names[i]), (self.nx,self.nz,self.length))
        #Stokes params
        if self.light_type == "Stokes params":
            original_stokes = self.charge_stokes_params(self.filename)
            original_stokes = np.memmap.reshape(inverse_scaling(original_stokes, self.light_scaler_name), (self.nx,self.nz,self.nlam,4))

        
        #Checking the path of directories is created
        dir_path = self.check_create_dirs("Images")
        print(dir_path)

        #Loading and plotting the predicted values vs the original ones

        for i in range(self.channels):
            ax[i].plot(original_stokes[:,iz,ilam,i], "r", label = "Generated")      
            ax[i].set_title(f"Stokes parameters spatial distribution -iz = {iz}, wavelength = {ilam}- title={self.light_title[i]}")
            ax[i].legend()
            ax[i].xaxis.set_major_locator(mticker.MultipleLocator(50))
            ax[i].xaxis.set_minor_locator(mticker.MultipleLocator(10))

        fig.savefig(dir_path + f"Stokes_parameter-{self.filename}.png")
        print(f"{self.filename} prediction plotted\n")
    def plot_predict_specific(self, filename, xz_coords, xz_titles):
        """
        xz_coords: coordinates per filename to plot
        """
        self.filename = filename
        
        #Load the predicted values stored in their respective directory
        predicted_values = np.load(f"{self.nn_model_type}/Predicted_values/{self.light_type}/obtained_value-{self.filename}.npy")
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.length,self.channels))
        #Charge the original parameters of the MURaM code
        original_atm = self.charge_atm_params(self.filename)
        original_atm = np.memmap.reshape(original_atm, (self.nx, self.nz, self.length,self.channels))
        for i in range(self.channels):
            original_atm[:,:,:,i] = np.memmap.reshape(inverse_scaling(original_atm[:,:,:,i], self.atm_scaler_names[i]), (self.nx,self.nz,self.length))
        
        #The values used in the original charge for the density where its log10 values, then here we invert that conversion.
        #predicted_values[:,:,:,2] = 10**predicted_values[:,:,:,i] 
        #original_atm[:,:,:,2] = 10**original_atm[:,:,:,i]

        #Checking the path of directories is created
        dir_path = self.check_create_dirs("Images")
        print(dir_path)

        #Plotting the predicted values vs the original ones
        N_plots = np.shape(xz_coords)[0]
        fig, ax = plt.subplots(N_plots,4,figsize=(50,10*N_plots))
        fig.suptitle("Atmosphere parameters", fontsize = 60)
        
        for j in range(N_plots):
            for i in range(self.channels):
                jtitle =xz_titles[j]
                ix = xz_coords[j][0]
                iz = xz_coords[j][1]
                ax[j,i].plot(range(self.length), predicted_values[ix,iz,:,i], label="Predicted curve")
                ax[j,i].set_title(f"{self.atm_title[i]}-({ix},{iz})-{jtitle}", fontsize = 30)
                ax[j,i].plot(range(self.length), original_atm[ix,iz,:,i], label="Original curve")
                ax[j,i].legend(fontsize = 20)
                ax[j,i].tick_params(axis = "both", labelsize = 20)
                ax[j,i].ticklabel_format(axis = "y", style = "sci", scilimits = (-3,3))
                ax[j,i].yaxis.get_offset_text().set_fontsize(24)
        print(f"""
        This is the path
        {dir_path}
        !!!!!!!!!!!!!!!!!!
        """)
        fig.savefig(dir_path + f"Atmosphere_parameter-{self.filename}.png")
        
        #Location plot
        stokes_params = self.charge_stokes_params(self.filename)
        fig, ax = plt.subplots()
        ax.set_title(f"Location {self.filename}", fontsize = 20)
        ax.imshow(stokes_params[:,:,0,0], cmap = "gist_gray")
        ax.scatter(xz_coords[:,0], xz_coords[:,1], color = "red")

        print(f"""
        This is the path
        {dir_path}
        !!!!!!!!!!!!!!!!!!
        """)
        fig.savefig(dir_path + f"location-{self.filename}.png")

        print(f"{self.filename} prediction plotted\n")

################################################################################################################
# LIGHT PREDICTING FEATURES
################################################################################################################


class LightTrainVisualMixin():
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
        self.output_ravel_shape = self.length*self.channels
        self.in_ls = (256-self.lb, 4)
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
                                                        verbose=1)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=[cp_callback])
        self.model.evaluate(self.te_input, self.te_output)
    def predict_values(self, filename):
        self.filename = filename
        self.charge_atm_params(filename)
        predicted_values = self.model.predict(np.memmap.reshape(self.atm_params, (self.nx*self.nz, 256-self.lb, 4)))
        predicted_values = np.memmap.reshape(predicted_values, (self.nx, self.nz, self.length, self.channels))
        #Inverse scaling application
        if self.light_type == "Intensity":
            predicted_values = np.memmap.reshape(inverse_scaling(predicted_values, self.scaler_name), (self.nx,self.nz))
        if self.light_type == "Stokes params":
            predicted_values = np.memmap.reshape(inverse_scaling(predicted_values, self.scaler_name), (self.nx,self.nz,self.length, self.channels))
        print(f"{filename} prediction done!")

        #Checking the path of directories is created
        dir_path = self.check_create_dirs("Predicted_values")

        #Saving the predicted values
        np.save(dir_path+f"obtained_value-{filename}.npy", predicted_values)
        return predicted_values
    def plot_predict(self, filename):
        ix = 200
        iz = 280
        lam = 10
        self.filename = filename
        #Loading and plotting the predicted values vs the original ones
        dir_path = self.check_create_dirs("Predicted_values")
        predicted_values = np.load(dir_path + f"obtained_value-{self.filename}.npy")
        
        #Checking the path of directories is created
        dir_path = self.check_create_dirs("Images")

        #Intensity
        if self.light_type == "Intensity":
            original_iout = self.charge_intensity(self.filename)
            original_iout = np.memmap.reshape(inverse_scaling(original_iout, self.scaler_name), (self.nx,self.nz))
            print(f"{self.filename} prediction done!")
            fig, ax = plt.subplots(1,4,figsize=(50,7))
            ax[0].imshow(predicted_values, cmap = "gist_gray")     
            ax[0].set_title(f"Intensity spatial distribution- title={self.title[i]}")
            ax[1].imshow(original_iout, cmap = "gist_gray")     
            ax[1].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")
            ax[2].imshow(np.abs(np.subtract(original_iout, predicted_values)), cmap = "gist_gray")     
            ax[2].set_title(f"Substraction of both images - title={self.title[i]}")
        
        #Stokes params
        if self.light_type == "Stokes parameters":
            original_stokes = self.charge_stokes_params(self.filename)
            original_stokes = np.memmap.reshape(inverse_scaling(original_stokes, self.scaler_name), (self.nx,self.nz,self.length,self.channels))
            print(f"{self.filename} prediction done!")
            fig, ax = plt.subplots(4,4,figsize=(50,7))


            for i in range(self.channels):
                ax[0,i].plot(range(self.length), predicted_values[ix,iz,:,i], label="Predicted curve")
                ax[0,i].set_title(f"Stokes parameters wavelength serie - title={self.title[i]} - ix={ix}, iy={iz}")
                ax[0,i].plot(range(self.length), original_stokes[ix,iz,:,i], label="Original curve")
                ax[0,i].legend()
                ax[1,i].plot(predicted_values[ix,:,lam,i], "k")     
                ax[1,i].set_title(f"Stokes parameters spatial distribution-[iz = {iz}] title={self.title[i]}")
                ax[1,i].plot(original_stokes[ix,:,lam,i], "k")     
                ax[1,i].set_title(f"ORIGINAL spatial distribution - title={self.title[i]}")

            fig.savefig(dir_path + f"Atmosphere_parameter-{self.filename}.png")
        print(f"{self.filename} prediction plotted\n")


