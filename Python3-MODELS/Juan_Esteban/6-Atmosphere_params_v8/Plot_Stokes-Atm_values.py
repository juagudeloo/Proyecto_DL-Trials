from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path
from train_generate.data_class import inverse_scaling

def main():
    ptm = path()
    atm_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    fln = "175000"
    atm_model.charge_atm_params(fln)
    orig_atm = np.load(f"/girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/6-Atmosphere_params_v8/atm_NN_model/Predicted_values/Stokes_params/obtained_value-{fln}.npy")
    for i in range(self.channels):
            original_atm[:,:,:,i] = np.memmap.reshape(inverse_scaling(original_atm[:,:,:,i], atm_model.scaler_names[i]), (atm_model.nx,atm_model.nz,atm_model.length))