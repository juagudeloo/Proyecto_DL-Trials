import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_NN_model

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = "53000"
    IN_LS = np.array([4,256]) #input shape in input layer
    model = Data_NN_model("Intensity", IN_LS, 1)
    model.model_train(tr_filename, TR_S = 0.75, epochs = 10)
    model.plot_loss()

    pr_filename = "56000"

    model.predict_values(pr_filename)
    model.plot_predict()
if __name__ == "__main__":
    main()