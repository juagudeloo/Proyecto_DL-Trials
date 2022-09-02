import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_class import Data_class, inverse_scaling
from pickle import dump, load

#
class NN_model_atm(Data_class):
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
        output = tf.keras.layers.Dense(4*(256-self.lb), activation=tf.nn.relu)

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
    def train(self,filename, tr_s, batch_size, epochs=8):
        """
        tr_s: training size percentage
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.split_data(filename, self.input_type, tr_s)
        self.history = self.model.fit(self.tr_input, self.tr_output[:,], epochs=self.epochs, batch_size=self.batch_size, verbose=1)
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
        filehandler = open("trained_model.pkl", "wb")
        dump(self.model, filehandler)
    ##### PREDICTING PHASE #####
    def load_model(self):
        filehandler = open("trained_model.pkl", "rb")
        self.model = load(filehandler)
    def predict_values(self, filename):
        self.pred_filename = filename
        print(f"{self.pred_filename} predicting...")
        if self.input_type == "Intensity":
            self.charge_intensity(self.pred_filename)
            self.predicted_values = self.model.predict(self.iout)
        if self.input_type == "Stokes params":
            self.charge_stokes_params(self.pred_filename)
            self.predicted_values = self.model.predict(self.profs)
        self.predicted_values = np.memmap.reshape(self.predicted_values, (self.nx, self.nz, 4, (256-self.lb)))
        scaler_names = ["mbyy", "mvyy", "mrho", "mtpr"]
        for i in range(len(scaler_names)):
            self.predicted_values[:,:,i,:] = np.memmap.reshape(inverse_scaling(self.predicted_values[:,:,i,:], scaler_names[i]), (self.nx,self.nz,(256-self.lb)))
        print(f"{self.pred_filename} prediction done!")
        np.save(f"obtained_values.npy-file={filename}-epochs={self.epochs}-batch_size={self.batch_size}.npy")
        return self.predicted_values
    def plot_predict(self):
        N_profs = 4
        ix = 200
        iz = 280
        height = 10
        title = ['Magnetic Field','Velocity','Density','Temperature']
        fig, ax = plt.subplots(4,4,figsize=(50,7))
        original_atm = self.charge_atm_params(self.pred_filename)
        original_atm = np.memmap.reshape(original_atm, (self.nx, self.nz, 4, (256-self.lb)))
        scaler_names = ["mbyy", "mvyy", "mrho", "mtpr"]
        for i in range(len(scaler_names)):
            original_atm[:,:,i,:] = np.memmap.reshape(inverse_scaling(original_atm[:,:,i,:], scaler_names[i]), (self.nx,self.nz,(256-self.lb)))
        print(f"{self.pred_filename} prediction done!")
        for i in range(N_profs):
            ax[0,i].plot(range(256-self.lb), self.predicted_values[ix,iz,i,:], label="Predicted curve")
            ax[0,i].set_title(f"Atmosphere parameters height serie - title={title[i]} - ix={ix}, iy={iz}")
            ax[0,i].plot(range(256-self.lb), original_atm[ix,iz,i,:], label="Original curve")
            ax[0,i].legend()
            ax[1,i].imshow(self.predicted_values[:,:,i,height], cmap = "gist_gray")     
            ax[1,i].set_title(f"Atmosphere parameters spatial distribution- title={title[i]}")
            ax[2,i].imshow(original_atm[:,:,i,height], cmap = "gist_gray")     
            ax[2,i].set_title(f"ORIGINAL spatial distribution - title={title[i]}")
            ax[3,i].imshow(np.abs(np.subtract(original_atm[:,:,i,height],self.predicted_values[:,:,i,height])), cmap = "gist_gray")     
            ax[3,i].set_title(f"Substraction of both images - title={title[i]}")

        if self.input_type == "Intensity":
            fig.savefig(f"Images/Intensity/Atmosphere_parameter-{self.filename}.png")
        if self.input_type == "Stokes params":
            fig.savefig(f"Images/Stokes_params/Atmosphere_parameter-{self.filename}.png")
        print(f"{self.pred_filename} prediction plotted\n")
