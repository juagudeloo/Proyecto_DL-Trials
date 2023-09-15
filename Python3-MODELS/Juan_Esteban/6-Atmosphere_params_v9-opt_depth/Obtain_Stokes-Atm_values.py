from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path_UIS


def main():
    ptm = path_UIS()
    atm_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    fln = "150000"
    atm_model.charge_stokes_params(fln, scale = True)
    print("shape after class:", atm_model.profs.shape)
    ##Model training
    #atm_model.compile_model()
    #atm_model.load_weights(f"{atm_model.nn_model_type}/training/{atm_model.light_type}/cp.ckpt")
    ##Model predicting
    #pr_filename = []
    #for i in np.arange(175, 200, 2):
    #    if i <100:
    #        a = "0"+str(i)+"000"
    #        pr_filename.append(a)
    #    else:
    #        a = str(i)+"000"
    #        pr_filename.append(a)
    #
    #print(atm_model.model.summary())
    #for fln in pr_filename:
    #    atm_model.predict_values(fln)
    #    atm_model.plot_predict(fln)


if __name__ == "__main__":
    main()
