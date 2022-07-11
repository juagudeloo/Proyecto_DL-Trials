import numpy as np
from data_class import Data_NN_model

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    filename = []
    for i in range(53,254):
        fln = "0"+str(i)+"000"
        filename.append(fln)
        developing_model(ptm, fln)
        

def developing_model(ptm, filename):
    model = Data_NN_model()
    model.charge_inputs(ptm, filename)

if __name__ == "__main__":
    main()
