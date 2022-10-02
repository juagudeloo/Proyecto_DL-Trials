from train_generate.extract_compile.data_class import Data_class
import numpy as np

def main():
    filename = "098000"
    sun = Data_class
    atm_params = sun.charge_atm_params(filename)
    intensity = sun.charge_intensity(filename)
    stokes = sun.charge_stokes_params
    print(np.shape(atm_params))
    print(np.shape(intensity))
    print(np.shape(stokes))


if __name__ == "__main__":
    main()
