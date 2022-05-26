from socket import ntohl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
#NN MODEL TESTING
class NN_MODEL:
    def __init__(self, nx, ny, nz, n_layers = 4, epochs = 20):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.epochs = epochs
        self.n_layers = n_layers #number of layers of the convolution model
        self.model = tf.keras.Sequential()
    def model_fitting(self, IN_LS, TR_D, TR_L, TR_BATCH_SIZE:float, TE_D, TE_L):
        if(self.n_layers == 1):
            self.model.add(tf.keras.layers.Input(shape = IN_LS, name='data_in'))
            self.model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
        if(self.n_layers == 2):
            self.model.add(tf.keras.layers.Input(shape = IN_LS, name='data_in'))
            self.model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
            self.model.add(tf.keras.layers.Conv1D(256, 2, activation='relu'))
        if(self.n_layers == 3):
            self.model.add(tf.keras.layers.Input(shape = IN_LS, name='data_in'))
            self.model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
            self.model.add(tf.keras.layers.Conv1D(256, 2, activation='relu'))
            self.model.add(tf.keras.layers.Conv1D(128, 2, activation='relu'))
        if(self.n_layers == 4):
            self.model.add(tf.keras.layers.Input(shape = IN_LS, name='data_in'))
            self.model.add(tf.keras.layers.Conv1D(512, 2, activation='relu'))
            self.model.add(tf.keras.layers.Conv1D(256, 2, activation='relu'))
            self.model.add(tf.keras.layers.Conv1D(128, 1, activation='relu'))
            self.model.add(tf.keras.layers.Conv1D(64, 2, activation='relu'))
        self.model.add(tf.keras.layers.GlobalMaxPool1D())
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.3)) #Layer added to avoid the overfitting
        self.model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
        opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
        self.history = self.model.fit(TR_D, TR_L, epochs=self.epochs, batch_size=TR_BATCH_SIZE, verbose=1)
        start_time = time.time()
        #TESTING
        print('*TEST*')
        metrics = self.model.evaluate(TE_D, TE_L)
        print("%s: %2.f%%" % (self.model.metrics_names[1], metrics[1]*100))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')
        print('\n CONV NEURAL TYPES DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
    #PREDICTING FUNCTION
    def predict_intensity(self, PR_D:float, pr_BATCH_SIZE:float):
        print("*predicting intensity...*")
        self.intensity_out = self.model_fitting().model.predict(PR_D, batch_size=pr_BATCH_SIZE, verbose=1)
        self.intensity_out = self.intensity_out.reshape(self.nx, self.nz)
        print("*intensity_done*\n")
        print(f"predicted intensity shape = {np.shape(self.intensity_out)}")
    #AFTER-TRAINING FUNCTIONS
    def plot_dist(self, PR_L, dist_title, dist_fig_name, loss_metric = 'mean_squared_error'):
        """
        ----------------------------------------------------------------------------
        Distribution and error relation plots
        ----------------------------------------------------------------------------
        intensity_out: array of the intensity predicted by the models
        history: array with the time information the training
        plot_var_values: 
        var_values: like the different values of the learning rate
        var_metrics: list with the time metrics of the testing step
        titles: list of strings with the title names of the distribution plots
        PR_L: Comparison data for the predicting values 
        dist_fig_name: Distribution plot file name
        error_fig_name: Error relation file name
        """
        path = "Images/"
        print("*plotting...*")
        diff = 0
        fig1, ax1 = plt.subplots(2, figsize = (40,10))
        diff = np.absolute(np.ravel(self.intensity_out)-np.ravel(PR_L))/np.absolute(np.ravel(PR_L))
        ax1[0].hist(x=diff, bins = 'auto',
                                    alpha=0.7, 
                                    rwidth=0.85)
        ax1[0].set_title(dist_title)
        ax1[1].imshow(self.intensity_out, cmap = 'gist_gray')
        ax1[1].set_title(dist_title)
        x = np.arange(1, self.epochs)+1
        y = self.history.history[loss_metric][1:10]
        print(np.size(y))
        print(np.size(x))
        ax1[2].scatter(x, y)
        ax1[2].set_ylabel(loss_metric)
        ax1[2].set_xlabel('epoch')
        ax1[2].set_ylim(0,0.3)
        fig1.savefig(path+dist_fig_name)
        print("*figure saved*")

