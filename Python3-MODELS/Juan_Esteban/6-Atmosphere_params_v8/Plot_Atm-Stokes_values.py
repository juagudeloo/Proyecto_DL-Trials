from train_generate.data_class import DataClass
import numpy as np
from path import path_UIS
from train_generate.data_class import inverse_scaling

def main():
    ptm = path_UIS()
    muram= DataClass(ptm = ptm)
    filename = "175000"

    #Charging the original data
    stokes_origin = muram.charge_stokes_params(filename=filename)
    atm_origin = muram.charge_atm_params(filename=filename)

    #Charging the obtained data
    stokes_obt = np.load("light_NN_model/Predicted_values/Stokes params/obtained_value-175000.npy")
    atm_obt = np.load("atm_NN_model/Predicted_values/Stokes params/obtained_value-175000.npy")


if __name__ == "__main__":
    main()
