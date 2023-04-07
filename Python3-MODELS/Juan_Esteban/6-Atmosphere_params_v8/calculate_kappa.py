import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from path import path_LOCAL, path_UIS
from train_generate.data_class import DataClass
from scipy.integrate import simps


def main():
    #path to kappa.0 file
    ptm = path_UIS()
    kappa_df = pd.read_table(ptm+"kappa.0.dat", sep=" ", header=None, skipinitialspace=True).dropna(axis=1, how = "all")

    #Temperature and Pressure values of the training grid
    tab_T=np.array([3.32, 3.34, 3.36, 3.38, 3.40, 3.42, 3.44, 3.46, 3.48, 3.50,
    3.52, 3.54, 3.56, 3.58, 3.60, 3.62, 3.64, 3.66, 3.68, 3.70,
    3.73, 3.76, 3.79, 3.82, 3.85, 3.88, 3.91, 3.94, 3.97, 4.00,
    4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50,
    4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00,
    5.05, 5.10, 5.15, 5.20, 5.25, 5.30 ])

    tab_p=np.array([-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 2.5,
        3., 3.5, 4., 4.5, 5., 5.5, 6. ,6.5, 7., 7.5, 8.])

    #Selecting the positions in (T, P) corresponding to each kappa value.
    #Remember that this values are the base 10 logarithm of the original ones.
    T = np.take(tab_T, kappa_df[0])
    P = np.take(tab_p, kappa_df[1])

    #Organizing the training points and its values
    TP_points =list(zip(T, P))
    kappa = kappa_df[2].to_numpy()

    #Training the linear interpolator
    f = LinearNDInterpolator(TP_points, kappa)
    
    #Charging the values of Temperature and Pressure from a snapshot of the MURAM simulation
    filename = "175000"
    kappa_C = KappaClass(ptm)
    kappa_C.charge_TP(filename)
    
    #finding the base 10 logarithm of the snapshot values
    T_muram = np.log10(kappa_C.mtpr[:,:,:])
    P_muram = np.log10(kappa_C.mprs[:,:,:])

    #Obtaining the corresponding inteporlated values of the MURAM snapshot
    kappa_cube = f(T_muram, P_muram)
    kappa_cube = np.multiply(kappa_cube, kappa_C.mrho)

    # Array for y distance in meters values (from 0 to 2560 in dy = 10 cm steps)
    Y = np.arange(0,256*10,10.)

    #Creation of the array to store the optical depth values
    opt_depth = np.zeros((kappa_C.nx,kappa_C.ny,kappa_C.nz))

    #Calculating the optical depth with the kappa integral
    for ix in range(kappa_C.nx):
        for iz in range(kappa_C.nz):
            for iy in range(kappa_C.ny):
                #The first value of the top 
                if iy == 0:
                    opt_depth[ix,kappa_C.ny-1,iz] = np.log10(kappa_cube[ix,kappa_C.ny-1:,iz])
                else:
                    print(len(kappa_cube[ix,kappa_C.ny-1-iy:,iz]))
                    print(len(Y[kappa_C.ny-1-iy:]))
                    a = simps(kappa_cube[ix,kappa_C.ny-1-iy:,iz]*, x = None, dx = 10)
                    # Base 10 logarithm of the original optical depth
                    opt_depth[ix,kappa_C.ny-1-iy,iz] = np.log10(a)
    np.save(f"optical_depth_{filename}.npy", opt_depth)

    IX = 100
    IZ = 100
    height = 180
    fig, ax = plt.subplots(1,2,figsize=(7,7))
    ax[0].imshow(opt_depth[:,height,:])
    ax[1].plot(Y, opt_depth[IX,:,IZ])
    fig.savefig("optical_depth_slice.pdf")
    

class KappaClass():
    def __init__(self, ptm, nx = 480, ny = 256, nz = 480):
        #size of the cubes of the data
        self.ptm = ptm
        self.nx = nx
        self.ny = ny
        self.nz = nz
    def charge_TP(self, filename):
        self.filename = filename
        print(f"reading EOS {self.filename}")
        #Charging temperature data
        self.EOS = np.memmap(self.ptm+"eos."+self.filename,dtype=np.float32)
        self.EOS = np.memmap.reshape(self.EOS, (2,self.nx,self.ny,self.nz), order="A")
        
        self.mrho = np.memmap(self.ptm+"result_0."+self.filename,dtype=np.float32)
        self.mtpr = self.EOS[0,:,:,:] 
        self.mprs = self.EOS[1,:,:,:]

        # n_eos -> 0: temperature ; 1: pressure

if __name__ == "__main__":
    main()