from train_generate.nn_model import AtmObtainModel
import numpy as np

def main():
    ptm = "/media/hdd/PRINCIPAL-2022-2/PROYECTOS/PROYECTO_DL/MURAM_data/"
    atm_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)

    #Model training
    atm_model.compile_model()
    atm_model.load_weights(f"{atm_model.nn_model_type}/training/{atm_model.light_type}/cp.ckpt")
    #Model predicting
    pr_filename = ["154000"]
    #for i in np.arange(83,100, 2):
    #    if i <100:
    #        a = "0"+str(i)+"000"
    #        pr_filename.append(a)
    #    else:
    #        a = str(i)+"000"
    #        pr_filename.append(a)
    
    
    for fln in pr_filename:
        atm_model.predict_values(fln)
        atm_model.plot_predict()


if __name__ == "__main__":
    main()
