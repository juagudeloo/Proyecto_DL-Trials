from path import path, low_boundary, top_boundary
from train_generate.nn_model import LightObtainModel
import numpy as np

def main():
    ptm = path()
    muram = LightObtainModel(ptm = ptm, low_boundary = low_boundary(), top_boundary = top_boundary())
    fln = "175000"
    muram.split_data_light_output(filename = fln,  light_type = "Stokes params", TR_S = 0.7)
    print(muram.in_ls)

if __name__ == "__main__":
    main()