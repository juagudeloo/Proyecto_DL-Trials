from path import path_UIS
from train_generate.data_class import DataClass
import numpy as np

def main():
    ptm = path_UIS()
    muram = DataClass(ptm = ptm)
    fln = "175000"
    muram.split_data_light_output(filename = fln)
    print(muram.in_ls)

if __name__ == "__main__":
    main()