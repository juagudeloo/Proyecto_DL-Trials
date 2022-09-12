import numpy as np
from nn_model import NN_model_atm
import time

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    non_existing = [85, 88, 89, 94, 95, 98]
    for i in np.arange(81,100,3):
        if i in non_existing:
            None
        else:
            a = "0"+str(i)+"000"
            tr_filename.append(a)
    IN_LS = np.array([256-180,4]) #input shape in input layer - Stokes profiles as input
    #Model training
    sun_model = NN_model_atm("Stokes params", create_scaler=False)
    sun_model.compile_model(IN_LS, learning_rate=0.001)
    
    start_time = time.time() #Time measured in seconds
    for fln in tr_filename:
        sun_model.train(fln, "training_1/cp.ckpt", tr_s = 0.75, batch_size= 1000, epochs=10)
        sun_model.plot_loss()

    with open("training_1/training_time.txt", "w") as f:
        f.write(f"{(time.time()-start_time)*(1/3600)} hours")

if __name__ == "__main__":
    main()