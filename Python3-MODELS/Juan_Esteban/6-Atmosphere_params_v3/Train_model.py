import numpy as np
from nn_model import NN_model_atm

def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    for i in np.arange(53,99,2):
        a = "0"+str(i)+"000"
        tr_filename.append(a)
    IN_LS = np.array([300,4]) #input shape in input layer - Stokes profiles as input
    #Model training
    sun_model = NN_model_atm("Stokes params", create_scaler=False)
    sun_model.compile_model(IN_LS, learning_rate=0.001)
    for fln in tr_filename:
        sun_model.train(fln, tr_s = 0.75, batch_size= 1000, epochs=100)
        sun_model.plot_loss()
    sun_model.save_model()
    #Model predicting
    pr_filename = []
    for i in range(59,61):
        a = "0"+str(i)+"000"
        pr_filename.append(a)
    
    for fln in pr_filename:
        sun_model.predict_values(fln)
        sun_model.plot_predict()


if __name__ == "__main__":
    main()
