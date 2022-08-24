import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_class

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    for i in range(53,58):
        a = "0"+str(i)+"000"
        tr_filename.append(a)
    #Model training
    dc = Data_class()
    dc.charge_atm_params(tr_filename[0])


if __name__ == "__main__":
    main()
