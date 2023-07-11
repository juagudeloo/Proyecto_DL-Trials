from path import path
from train_generate.data_class import DataClass
import numpy as np

def main():
    ptm = path()
    muram = DataClass(ptm = ptm)
    fln = "175000"
    muram.split_data_light_output(filename = fln,  light_type = "Stokes params", TR_S = 0.7)
    print(muram.in_ls)

if __name__ == "__main__":
    main()