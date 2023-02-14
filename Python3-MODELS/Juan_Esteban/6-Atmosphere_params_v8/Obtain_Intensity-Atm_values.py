from train_generate.nn_model import AtmObtainModel
import numpy as np
from path import path

def main():
    ptm = path()
    atm_model = AtmObtainModel(ptm = ptm, light_type="Intensity", create_scaler=False)

    #Model training
    atm_model.compile_model()
    atm_model.load_weights(f"{atm_model.nn_model_type}/training/{atm_model.light_type}/cp.ckpt")
    #Model predicting
    pr_filename = []
    for i in np.arange(175,200, 2):
        if i <100:
            a = "0"+str(i)+"000"
            pr_filename.append(a)
        else:
            a = str(i)+"000"
            pr_filename.append(a)
    
    for fln in pr_filename:
        atm_model.predict_values(fln)
        atm_model.plot_predict()


if __name__ == "__main__":
    main()
