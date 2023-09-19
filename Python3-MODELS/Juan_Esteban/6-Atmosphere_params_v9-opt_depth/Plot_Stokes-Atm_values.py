from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path_UIS
from train_generate.data_class import inverse_scaling

def main():
    ptm = path_UIS()
    atm_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)

    #Model training
    #atm_model.compile_model()
    #atm_model.load_weights(f"{atm_model.nn_model_type}/training/{atm_model.light_type}/cp.ckpt")
    #Model predicting
    pr_filename = []
    #for i in np.arange(175, 200, 2):
    #    if i <100:
    #        a = "0"+str(i)+"000"
    #        pr_filename.append(a)
    #    else:
    #        a = str(i)+"000"
    #        pr_filename.append(a)
    fln = "175000"
    atm_model.plot_predict(fln)
    #for fln in pr_filename:
    #    atm_model.plot_predict(fln)

if __name__ == "__main__":
    main()
