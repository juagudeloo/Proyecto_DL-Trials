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
    #Model training
    sun_model = NN_model("Stokes params")
    sun_model.compile_model(IN_LS)
    for fln in tr_filename:
        sun_model.train(fln, tr_s = 0.75, batch_size_percentage = 0.05, epochs=3)
        sun_model.plot_loss()
    #Model predicting
    pr_filename = []
    for i in range(61,70):
        a = "0"+str(i)+"000"
        pr_filename.append(a)
    
    for fln in pr_filename:
        sun_model.predict_values(fln)
        sun_model.plot_predict()

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
    def train(self,filename, tr_s, batch_size_percentage, epochs=8):
        """
        batch_size: its a fraction relative to the total of the set (must be between 0<x<1).
        """
        self.split_data(filename, self.output_type, tr_s)
        TR_BATCH_SIZE = int(self.tr_input[:,1,2].size*batch_size_percentage)
        self.history = self.model.fit(self.tr_input, self.tr_output, epochs=epochs, batch_size=TR_BATCH_SIZE, verbose=1)
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
            fig, ax = plt.subplots(2,4,figsize=(7,28))
            for i in range(N_profs):
                ax[0,i].plot(range(self.nlam), self.predicted_values[ix,iz,i,:])
                ax[0,i].set_title(f"Stokes params spectra - title={title[i]} - ix={ix}, iy={iz}")
                ax[1,i].imshow(self.predicted_values[:,:,i,wave_lam])     
                ax[1,i].set_title(f"Stokes params spatial distribution- title={title[i]}")
            fig.savefig(f"Images/Stokes_params/Predicted_Stokes_parameters-{self.filename}.png")   
        print(f"{self.filename} prediction plotted\n")
if __name__ == "__main__":
    main()