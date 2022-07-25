from tkinter.tix import InputOnly
import matplotlib.pyplot as plt
import numpy as np
from nn_model import NN_model_indiv_atm

def main():
    physical_magnitudes = ["mbyy", "mvyy", "mrho", "mtpr"]
    for phys_m in physical_magnitudes:
        physical_magnitude_fitting(phys_m)

def physical_magnitude_fitting(phys_m):
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    for i in range(53,56):
        a = "0"+str(i)+"000"
        tr_filename.append(a)
    input_type = "Stokes params"
    if input_type == "Intensity":
        IN_LS = np.array([1]) #input shape in input layer - Stokes profiles as input
    elif input_type == "Stokes params":
        IN_LS = np.array([4,300]) #input shape in input layer - Stokes profiles as input
    #Model training
    sun_model = NN_model_indiv_atm(input_type, phys_m)
    sun_model.compile_model(IN_LS)
    for fln in tr_filename:
        sun_model.train(fln, tr_s = 0.75, batch_size= 10000, epochs=3)
        sun_model.plot_loss()
    #Model predicting
    pr_filename = []
    for i in range(56,58):
        a = "0"+str(i)+"000"
        pr_filename.append(a)
    for fln in pr_filename:
        sun_model.predict_values(fln)
        sun_model.plot_predict()

if __name__ == "__main__":
    main()