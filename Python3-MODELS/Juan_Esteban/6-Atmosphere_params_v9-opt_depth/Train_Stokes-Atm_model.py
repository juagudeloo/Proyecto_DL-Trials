import numpy as np
from train_generate.nn_model import AtmObtainModel
import time
from path import path_UIS
import sys

def main():
    #Intensity specifications
    ptm = path_UIS()
    fln = str(sys.argv[1])
    print(f"###################### filename", fln)
    
    stokes_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    stokes_model.compile_model(learning_rate=0.001)
    checkpoint_path = "/girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/6-Atmosphere_params_v9-opt_depth/atm_NN_model/training/Stokes params/cp.ckpt"
    stokes_model.load_weights(checkpoint_path)

    start_time = time.time() #Time measured in seconds
    stokes_model.train(fln, tr_s = 0.75, batch_size= 1000, epochs=40)
    stokes_model.plot_loss()

    with open(f"{stokes_model.nn_model_type}/training/training_time_{fln}.txt", "w") as f:
        f.write(f"{(time.time()-start_time)*(1/3600)} hours")

if __name__ == "__main__":
    main()