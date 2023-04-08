from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path_UIS
from train_generate.data_class import inverse_scaling

def main():
    ptm = path_UIS()
    atm_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    filenames = []
    xz_coords = {}
    xz_titles = []
    for num in np.arange(175,201,2):
        fln = str(num)+"000"
        filenames.append(fln)
        xz_coords[fln] = []

    iz = 280
    xz_coords["175000"] = np.array([[48,iz],
                                     [220,iz],
                                     [250,iz],
                                     [340,iz]])
    xz_titles["175000"] = ["intergranular",
                           "granular",
                           "intergranular",
                           "granular"]

    xz_coords["177000"] = np.array([[30,iz],
                                     [120,iz],
                                     [252,iz],
                                     [300,iz]])
    xz_titles["177000"] = ["granular",
                           "granular",
                           "intergranular",
                           "granular"]


    for fln in filenames:
        ## Plot for checking the stokes parameters spatial distribution for a given value of ix or iz

    	#atm_model.plot_predict_initial(fln)

        ## Plot the atmosphere params for given ix and iz values

        atm_model.plot_predict_specific(fln, xz_coords[fln], xz_titles[fln])

if __name__ == "__main__":
    main()
