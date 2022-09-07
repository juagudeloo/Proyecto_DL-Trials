import matplotlib.pyplot as plt
import numpy as np

def main():
    obtained_file = int(input("File of the obtained atmosphere values: "))
    atm_params = np.load(f"obtained_value-0{obtained_file}000.npy")
    height = 10
    max_values = {}
    min_values = {}
    for i in range(4):
        max_values[i] = np.argwhere(atm_params[:,:,i,height]==np.max(atm_params[:,:,i,height]))
        min_values[i] = np.argwhere(atm_params[:,:,i,height]==np.min(atm_params[:,:,i,height]))
    
    for i in range(4):
        print(atm_params[max_values[i][0][0], max_values[i][0][1], i, height])



if __name__ == "__main__":
    main()