from nn_model import NN_model_atm
import numpy as np

def main():
    sun_model = NN_model_atm("Stokes params", create_scaler=False)
    sun_model.load_model()
    #Model predicting
    pr_filename = []
    non_existing = [85, 88, 89, 94, 95, 98]
    for i in np.arange(83,100, 2):
        if i in non_existing:
            None
        else:
            a = "0"+str(i)+"000"
            pr_filename.append(a)
    
    for fln in pr_filename:
        sun_model.predict_values(fln)
        sun_model.plot_predict()


if __name__ == "__main__":
    main()