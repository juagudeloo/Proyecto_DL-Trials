from train_generate.data_class import DataClass
from path import path_UIS()
import numpy as np

def main():

    ptm = path_UIS()
    muram = DataClass(ptm)
    intensities = []

    for i in np.arange(53*1000,(200+1)*1000,1000):
        fln = str(i)
        intensities.append(muram.charge_intensity())

    intensities = np.array(intensities)


if __main__ == "__name__":
    main()
