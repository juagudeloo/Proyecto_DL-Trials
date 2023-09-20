import numpy as np
from train_generate.nn_model import LightObtainModel
import time
from path import path_UIS
import sys

def main():
    #Intensity specifications
    ptm = path_UIS()
    tr_filename = []
    tr_filename.append("175000")
    #for i in np.arange(80,150+1,7):
    #    if i==98:
    #        None
    #    else:
    #        if i<100:
    #            a = "0"+str(i)+"000"
    #            tr_filename.append(a)
    #        else:
    #            a = str(i)+"000"
    #            tr_filename.append(a)

    atm_model = LightObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    atm_model.compile_model(learning_rate=0.001)
    
    start_time = time.time() #Time measured in seconds
    print("is running")
    for fln in tr_filename:
        atm_model.train(fln, tr_s = 0.75, batch_size= 10000, epochs=40)
        atm_model.plot_loss()
    print(atm_model.model.summary())

    with open(f"{atm_model.nn_model_type}/training/training_time.txt", "w") as f:
        f.write(f"{(time.time()-start_time)*(1/3600)} hours")

if __name__ == "__main__":
    main()