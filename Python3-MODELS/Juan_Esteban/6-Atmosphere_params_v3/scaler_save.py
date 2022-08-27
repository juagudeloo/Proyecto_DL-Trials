import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_class

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    names = ["mbyy", "mvyy", "mrho", "mtpr", "iout", "stokes"]
    scaler_pairs = {names[0]: [],
                     names[1]: [],
                     names[2]: [],
                     names[3]: [],
                     names[4]: [],
                     names[5]: []}
    dc = Data_class()
    for i in range(53,254):
        a = "0"+str(i)+"000"
        tr_filename.append(a)
        dc.charge_atm_params(a)
        dc.charge_intensity(a)
        dc.charge_stokes_params(a)
        for nm in names:
            scaler_pairs[nm].append(np.load(nm+".npy"))
    
    print(scaler_pairs)
    



if __name__ == "__main__":
    main()
