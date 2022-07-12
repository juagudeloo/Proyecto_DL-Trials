import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_NN_model

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = "053000"
    IN_LS = (4,256) #input shape in input layer
    model = Data_NN_model("Intensity")
    model.model_train(tr_filename, TR_S = 0.75, IN_LS = IN_LS, OUT_LS = 1, epochs = 10)
    model.plot_loss()

    pr_filename = "056000"

    model.predict_values(pr_filename)
    model.plot_predict()
if __name__ == "__main__":
    main()