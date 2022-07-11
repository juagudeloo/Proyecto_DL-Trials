import numpy as np
from data_class import Data_NN_model

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    filename = "056000"

    model = Data_NN_model()
    model.charge_inputs(ptm, filename)


if __name__ == "__main__":
    main()
