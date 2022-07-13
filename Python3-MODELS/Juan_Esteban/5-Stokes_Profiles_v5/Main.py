import matplotlib.pyplot as plt
import numpy as np


def main():
    #Intensity specifications
    ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"
    tr_filename = []
    for i in range(53,55):
        a = "0"+str(i)+"000"
        tr_filename.append(a)
    IN_LS = np.array([4,256]) #input shape in input layer
    #Model training
    sun_model = NN_model("Stokes params")
    sun_model.compile_model(IN_LS)
    for fln in tr_filename:
        sun_model.train(fln, tr_s = 0.75, batch_size_percentage = 0.05, epochs=3)
        sun_model.plot_loss()
    #Model predicting
    pr_filename = []
    for i in range(56,57):
        a = "0"+str(i)+"000"
        pr_filename.append(a)
    
    for fln in pr_filename:
        sun_model.predict_values(fln)
        sun_model.plot_predict()


if __name__ == "__main__":
    main()