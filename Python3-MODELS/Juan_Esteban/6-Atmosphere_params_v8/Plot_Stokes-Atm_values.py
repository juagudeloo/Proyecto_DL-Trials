from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path
from train_generate.data_class import inverse_scaling

def main():
    ptm = path()
    atm_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    fln = "175000"
    atm_model.plot_predict(fln)

if __name__ == "__main__":
    main()