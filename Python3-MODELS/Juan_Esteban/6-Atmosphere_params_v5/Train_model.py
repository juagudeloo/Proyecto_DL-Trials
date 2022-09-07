import numpy as np
from nn_model import NN_model_atm

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
    IN_LS = np.array([300,4]) #input shape in input layer - Stokes profiles as input
    #Model training
    sun_model = NN_model_atm("Stokes params", create_scaler=False)
    sun_model.compile_model(IN_LS, learning_rate=0.001)
    for fln in tr_filename:
        sun_model.train(fln, "training_1/cp.ckpt", tr_s = 0.75, batch_size= 1000, epochs=10)
        sun_model.plot_loss()

if __name__ == "__main__":
    main()