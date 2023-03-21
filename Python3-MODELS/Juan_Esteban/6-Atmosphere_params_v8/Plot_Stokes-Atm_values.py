from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path
from train_generate.data_class import inverse_scaling

def main():
    ptm = path()
    atm_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    filenames = []
    xz_coords = {}
    for num in np.arange(175,201,2):
        fln = str(num)+"000"
        filenames.append(fln)
        xz_coords[fln] = []


    for fln in filenames:
        ## Plot for checking the stokes parameters spatial distribution for a given value of ix or iz

    	atm_model.plot_predict_initial(fln)

        ## Plot the atmosphere params for given ix and iz values

        #atm_model.plot_predict_specific(fln, xz_coords[fln])

if __name__ == "__main__":
    main()
