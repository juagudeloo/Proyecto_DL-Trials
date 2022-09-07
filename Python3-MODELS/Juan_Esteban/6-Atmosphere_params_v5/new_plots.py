import matplotlib.pyplot as plt
import numpy as np

def main():
    obtained_file = int(input("File of the obtained atmosphere values: "))
    atm_params = np.load(f"obtained_value-0{obtained_file}000.npy")
    height = 10
    max_min_mbyy = np.argwhere(atm_params[:,:,0,height]==np.max(atm_params[:,:,0,height]))
    print(atm_params[max_min_mbyy[0], max_min_mbyy[1], 0, height])


if __name__ == "__main__":
    main()