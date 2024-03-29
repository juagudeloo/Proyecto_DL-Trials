import numpy as np
from train_generate.nn_model import LightObtainModel
import time
from path import path
import sys

def main():
    #Intensity specifications
    ptm = path()

    ifl = int(sys.argv[1])
    if ifl < 100:
        fln = "0"+str(ifl)+"000"
    else:
        fln = str(ifl)+"000"

    atm_model = LightObtainModel(ptm = ptm, light_type="Stokes params", create_scaler=False)
    atm_model.compile_model(learning_rate=0.001)
    
    start_time = time.time() #Time measured in seconds
    print("is running")
    atm_model.train(fln, tr_s = 0.75, batch_size= 10000, epochs=40)
    atm_model.plot_loss()
    print(atm_model.model.summary())

    with open(f"{atm_model.nn_model_type}/training/training_time.txt", "w") as f:
        f.write(f"{(time.time()-start_time)*(1/3600)} hours")

if __name__ == "__main__":
    main()