import numpy as np
from data_class import Data_NN_model

def main():
    n_file = ["053000", "054000"]
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    filename = n_file

    model = Data_NN_model()
    a,b = model.charge_inputs(ptm, filename)

if __name__ == "__main__":
    main()
