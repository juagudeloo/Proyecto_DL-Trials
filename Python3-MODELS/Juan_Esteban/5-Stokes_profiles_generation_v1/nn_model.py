import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_class import Data_class_Stokes, inverse_scaling
from pickle import dump, load
import os

#
class NN_model_atm(Data_class_Stokes):
    def __init__(self, input_type, nx = 480, ny = 256, nz = 480, lower_boundary=180, create_scaler = True):
        """
        lower_boundary -> indicates from where to take the data for training.
        output_type options:
        "Intensity" -> The model predicts intensity.
        "Stokes params" -> The model predicts the Stokes parameters.
        create_scaler -> Set True by default. It determines wheter to create a scaler object or take an already created one.
        """
        super().__init__(nx,ny,nz,lower_boundary, create_scaler)
        self.input_type = input_type
    def compile_model(self, in_ls, learning_rate=0.001):
        data_in =  tf.keras.layers.Input(shape = in_ls, name='data_in')
        conv1 = tf.keras.layers.Conv1D(512, kernel_size = 2, activation=tf.nn.relu)
        conv2 = tf.keras.layers.Conv1D(256, kernel_size = 2, activation=tf.nn.relu)
        conv3 = tf.keras.layers.Conv1D(64, kernel_size = 2, activation=tf.nn.relu) 
        dropout = tf.keras.layers.Dropout(0.5)
        flattened = tf.keras.layers.Flatten()  
        output = tf.keras.layers.Dense(4*self.nlam, activation=tf.nn.sigmoid)

        input = conv1(data_in)
        x = conv2(input)
        x = conv3(x)
        #x = dropout(x)
        x = flattened(x) #If this layer is not put, then the output will be of 4 channels......but for some reason is not working here
        x = output(x)

        self.model = tf.keras.models.Model(inputs = data_in, outputs = x)
        opt_func = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        self.model.summary()
        
        return self.model
    def train(self,filename, checkpoint_path, tr_s, batch_size, epochs=8, ):
        """
        tr_s: training size percentage
        """
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.split_data(filename, self.input_type, tr_s)
        self.history = self.model.fit(self.tr_input, self.tr_output[:,], epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=[cp_callback])
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
            fig.savefig(f"Images/Stokes_params/loss_plot-{self.filename}.png")
        print(f"{self.filename} loss plotted!")
    def save_model(self):
        filehandler = open(f"trained_model.pkl", "wb")
        dump(self.model, filehandler)
    ##### PREDICTING PHASE #####
    def load_model(self):
        filehandler = open("trained_model-epochs=10-batch_size=1000.pkl", "rb")
        self.model = load(filehandler)
    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)
    def predict_values(self, filename):
        self.pred_filename = filename
        print(f"{self.pred_filename} predicting...")
        self.charge_atm_params(self.pred_filename)
        self.predicted_values = self.model.predict(self.atm_params)
        self.predicted_values = np.memmap.reshape(self.predicted_values, (self.nx, self.nz, 4, self.nlam))
        if self.input_type == "Stokes params":
            self.predicted_values = np.memmap.reshape(inverse_scaling(self.predicted_values, "stokes"), (self.nx,self.nz,300,4))
            for i in range(1,4):
                self.predicted_values[:,:,:,i] = self.predicted_values[:,:,:,i]/self.predicted_values[:,:,:,0]
            print(f"{self.pred_filename} prediction done!")
            np.save(f"/mnt/scratch/juagudeloo/obtained_data/Stokes_obtained_values-{filename}.npy", self.predicted_values)
        return self.predicted_values
    def plot_predict(self):
        N_profs = 4
        ix = 200
        iz = 280
        wavelength = 200
        fig, ax = plt.subplots(4,4,figsize=(50,7))
        original_stokes = self.charge_stokes_params(self.pred_filename)
        ylabel = ["$I$ [ph]", "$Q/I$", "$U/I$", "$V/I$"]
        original_stokes = np.memmap.reshape(inverse_scaling(original_stokes, "stokes"), (self.nx,self.nz,self.nlam, 4))
        print(f"{self.pred_filename} prediction done!")
        for i in range(N_profs):
            ax[0,i].plot(np.arange(6302,6302+10*self.nlam, 10), self.predicted_values[ix,iz,:,i], label="Generated curve")
            ax[0,i].set_title(f"ix={ix}, iy={iz}")
            ax[0,i].plot(np.arange(6302,6302+10*self.nlam, 10), original_stokes[ix,iz,:,i], label="Original curve")
            ax[0,i].set_ylabel(ylabel[i])
            ax[0,i].set_xlabel(r"$\lambda$ [$\AA$]")
            ax[0,i].legend()

            ax[1,i].imshow(self.predicted_values[:,:,wavelength,i], cmap = "gist_gray")     
            ax[1,i].set_title(f"Generated spatial distribution- title={ylabel[i]}")

            ax[2,i].imshow(original_stokes[:,:,wavelength,i], cmap = "gist_gray")     
            ax[2,i].set_title(f"ORIGINAL spatial distribution - title={ylabel[i]}")

            ax[3,i].imshow(np.abs(np.subtract(original_stokes[:,:,wavelength,i],self.predicted_values[:,:,wavelength,i])), cmap = "gist_gray")     
            ax[3,i].set_title(f"Substraction of both images - title={ylabel[i]}")

        if self.input_type == "Intensity":
            fig.savefig(f"Images/Intensity/Atmosphere_parameter-{self.filename}.png")
        if self.input_type == "Stokes params":
            fig.savefig(f"Images/Stokes_params/Atmosphere_parameter-{self.filename}.png")
        print(f"{self.pred_filename} prediction plotted\n")
