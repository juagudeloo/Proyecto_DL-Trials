from train_generate.nn_model import LightObtainModel
import numpy as np
from path import path, low_boundary

def main():
    ptm = path()
    light_model = LightObtainModel(ptm = ptm, light_type="Stokes params", low_boundary=low_boundary(), top_boundary = top_boundary(), create_scaler=False)

    #Model training
    light_model.compile_model()
    light_model.load_weights(f"{light_model.nn_model_type}/training/{light_model.light_type}/cp.ckpt")
    #Model predicting
    pr_filename = []
    for i in np.arange(175,200, 2):
        if i <100:
            a = "0"+str(i)+"000"
            pr_filename.append(a)
        else:
            a = str(i)+"000"
            pr_filename.append(a)
    print(light_model.model.summary())
    
    for fln in pr_filename:
        light_model.predict_values(fln)
        light_model.plot_predict(fln)
    


if __name__ == "__main__":
    main()
