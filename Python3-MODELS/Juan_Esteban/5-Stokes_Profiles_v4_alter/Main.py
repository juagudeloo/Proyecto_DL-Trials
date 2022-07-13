import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_class import Data_class
import testing_functions as tef

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = "053000"

    data = Data_class()
    TR_D, TR_L, TE_D, TE_L = data.split_data(tr_filename, TR_S = 0.75, output_type="Intensity")

    IN_LS = np.array([4,256]) #input shape in input layer
    TR_BATCH_SIZE = int(len(TR_D[:,1,2])/1)

    model = tef.model_dense_layers(IN_LS, n_layers = 4)
    opt_func = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer = opt_func, metrics = [tf.keras.metrics.MeanSquaredError()])
    model.summary()
    model.fit(TR_D, TR_L, epochs=8, batch_size=TR_BATCH_SIZE, verbose=1)

    
if __name__ == "__main__":
    main()