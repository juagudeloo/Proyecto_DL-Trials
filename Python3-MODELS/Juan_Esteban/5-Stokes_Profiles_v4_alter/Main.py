import matplotlib.pyplot as plt
import numpy as np
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

    #MODELS DATA DICTIONARY
    models = []
    history = []
    metrics = []
    intensity_pred = []

    #PLOT INFORMATION
    titles = ['CNN 1', 'CNN 2', 'CNN 3', 'CNN 4']
                            
    dist_name = "nn_conv_models_dist.png"
    error_fig_name = "nn_conv_models_error.png"
    ##### NEURAL NETWORK TYPE VARIATIONS
    #dense models
    models, metrics, history = tef.test_conv_models(IN_LS, TR_D, TR_L, TR_BATCH_SIZE, TE_D, TE_L)

    
if __name__ == "__main__":
    main()