from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path, low_boundary, top_boundary


def main():
    ptm = path()
    atm_model = AtmObtainModel(ptm = ptm, opt_len = 5, light_type="Stokes params", create_scaler=False)

    #Model training
    TR_S = 0.8
    filename = "080000"
    light_type="Stokes params"
    atm_model.split_data_atm_output(filename, light_type, TR_S)
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
