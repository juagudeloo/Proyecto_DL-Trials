import matplotlib.pyplot as plt
import numpy as np
from data_class import Data_class
import pandas as pd

#
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
    for i in range(53,200):
        if i == 76:
            None
        elif i<100:
            a = "0"+str(i)+"000"
            tr_filename.append(a)
            dc.charge_atm_params(a)
            dc.charge_intensity(a)
            dc.charge_stokes_params(a)
            for nm in names:
                scaler_pairs[nm].append(np.load(nm+".npy"))
        elif i >= 100:
            a = str(i)+"000"
            tr_filename.append(a)
            dc.charge_atm_params(a)
            dc.charge_intensity(a)
            dc.charge_stokes_params(a)
            for nm in names:
                scaler_pairs[nm].append(np.load(nm+".npy"))
    for i in range(115,201):
        a = str(i)+"000"
        tr_filename.append(a)
        dc.charge_atm_params(a)
        dc.charge_intensity(a)
        dc.charge_stokes_params(a)
        for nm in names:
            scaler_pairs[nm].append(np.load(nm+".npy"))
    
    
    for nm in names:
        scaler_pairs[nm] = [np.min(scaler_pairs[nm]), np.max(scaler_pairs[nm])]
    scaler_pairs = pd.DataFrame(scaler_pairs)
    scaler_pairs.to_csv("scaler_pairs.csv", index=False)
    



if __name__ == "__main__":
    main()
