import numpy as np
from data_class import Data_NN_model

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    filename = []
    for i in range(53,253):
        if i < 100:
            fln = "0"+str(i)+"000"
            filename.append(fln)
        else:
            fln = str(i)+"000"
        developing_model(ptm, fln)
        

def developing_model(ptm, filename):
    model = Data_NN_model()
    model.charge_stokes_params(ptm, filename)

if __name__ == "__main__":
    main()
