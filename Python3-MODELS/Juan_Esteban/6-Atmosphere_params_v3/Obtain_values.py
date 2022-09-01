from nn_model import NN_model_atm

def main():
    sun_model = NN_model_atm("Stokes params", create_scaler=False)
    sun_model.load_model()
    #Model predicting
    pr_filename = []
    for i in range(120,140):
        a = "0"+str(i)+"000"
        pr_filename.append(a)
    
    for fln in pr_filename:
        sun_model.predict_values(fln)
        sun_model.plot_predict()


if __name__ == "__main__":
    main()
