from train_generate.data_class import DataClass
from path import path_UIS
import numpy as np

def main():
    ptm = path_UIS()
    muram = DataClass(ptm=ptm)

    for i in np.arange(97000, 223000, 1000):
        if 1 < 100000:
            fln = "0"+str(i)
        else:
            fln = str(i)
        muram.resave_stokes_params(fln)

if __name__ == "__main__":
    main()