import numpy as np
from train_generate.nn_model import LightObtainModel
import time

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    for i in np.arange(80,171,3):
        if i<100:
            a = "0"+str(i)+"000"
            tr_filename.append(a)
        else:
            a = str(i)+"000"
            tr_filename.append(a)
    atm_model = LightObtainModel(light_type="Stokes params", create_scaler=False)
    atm_model.compile_model(learning_rate=0.001)
    
    start_time = time.time() #Time measured in seconds
    for fln in tr_filename:
        atm_model.train(fln, tr_s = 0.75, batch_size= 1000, epochs=1)
        atm_model.plot_loss()

    with open(f"{atm_model.nn_model_type}/training/training_time.txt", "w") as f:
        f.write(f"{(time.time()-start_time)*(1/3600)} hours")

if __name__ == "__main__":
    main()