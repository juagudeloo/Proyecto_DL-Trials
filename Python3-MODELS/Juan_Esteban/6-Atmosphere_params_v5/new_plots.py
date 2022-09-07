import matplotlib.pyplot as plt
import numpy as np

def main():
    obtained_file = int(input("File of the obtained atmosphere values:"))
    atm_params = np.load(f"obtained_value-0{obtained_file}000.npy")
    print(np.shape(atm_params))


if __name__ == "__main__":
