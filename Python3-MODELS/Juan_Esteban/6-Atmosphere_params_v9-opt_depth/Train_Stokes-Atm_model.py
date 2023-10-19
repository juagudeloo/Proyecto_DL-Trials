import numpy as np
from train_generate.nn_model import AtmObtainModel
import time
from path import path_UIS
import sys

def main():
    #Intensity specifications
    ptm = path_UIS()
    tr_filename = []
    print(f"######################## filename{fln}")

    with open(f"{stokes_model.nn_model_type}/training/training_time_{fln}.txt", "w") as f:
        f.write(f"{(time.time()-start_time)*(1/3600)} hours")

    for i in np.arange(80,94+1,7):
        if i==98:
            None
        else:
            if i<100:
                a = "0"+str(i)+"000"
                tr_filename.append(a)
            else:
                a = str(i)+"000"
                tr_filename.append(a)
    
    stokes_model = AtmObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    stokes_model.compile_model(learning_rate=0.001)
    
    start_time = time.time() #Time measured in seconds
    for fln in tr_filename:
        stokes_model.train(fln, tr_s = 0.75, batch_size= 10000, epochs=40)
        stokes_model.plot_loss()

    with open(f"{stokes_model.nn_model_type}/training/training_time.txt", "w") as f:
        f.write(f"{(time.time()-start_time)*(1/3600)} hours")

if __name__ == "__main__":
    main()