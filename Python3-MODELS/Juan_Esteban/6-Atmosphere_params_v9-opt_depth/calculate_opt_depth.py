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
    
    #Charging the values of Temperature and Pressure from a snapshot of the MURAM simulation
    OD = OptDepthClass(ptm)
    for num in np.arange(52000,224000,1000):
        if num<100000:
            fln = "0"+str(num)
        else:
            fln = str(num)
        OD.opt_depth_calculation(filename=fln)
        
        
    

class OptDepthClass():
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

        self.mrho = np.memmap.reshape(self.mrho, (self.nx,self.ny,self.nz))
        self.mtpr = self.EOS[0,:,:,:] 
        self.mprs = self.EOS[1,:,:,:]
    def kappa_interpolation(self):
        kappa_df = pd.read_table(self.ptm+"kappa.0.dat", sep=" ", header=None, skipinitialspace=True).dropna(axis=1, how = "all")
        """
        The model used for this optical depth table is the the Rosseland opacity table
        """
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
        k = kappa_df[2].to_numpy()

        #Training the linear interpolator
        self.kappa = LinearNDInterpolator(TP_points, k)
    def opt_depth_calculation(self, filename):
        self.charge_TP(filename)
        self.kappa_interpolation()

        #finding the base 10 logarithm of the snapshot values
        T_muram = np.log10(self.mtpr[:,:,:])
        self.Tmin = 3.32
        self.Tmin_before = self.Tmin+0.0001
        self.Tmax = 5.30
        self.Tmax_before = self.Tmin-0.0001
        self.Pmin = -2.
        self.Pmin_before = self.Pmin+0.0001
        self.Pmax = 8.
        self.Pmax_before = self.Pmin-0.0001
        T_muram[T_muram <= self.Tmin] = self.Tmin_before #we bound the upper values to fit inside the domain of the atmosphere model and this is possible because 
                                                    #this points of the atmosphere does not affect in the creation of the FeI lines creation.
        T_muram[T_muram >= self.Tmax] = self.Tmax_before
        P_muram = np.log10(self.mprs[:,:,:])
        P_muram[P_muram <= self.Tmin] = self.Pmin_before #we bound the upper values to fit inside the domain of the atmosphere model and this is possible because 
                                                    #this points of the atmosphere does not affect in the creation of the FeI lines creation.
        P_muram[P_muram >= self.Tmax] = self.Pmax_before

        #Obtaining the corresponding inteporlated values of the MURAM snapshot.
        kappa_cube = self.kappa(T_muram, P_muram)
        #Kappa is originally normalized by density. Here we denormalize it.
        kappa_cube = np.multiply(kappa_cube, self.mrho)

        # Array for y distance in meters values (from 0 to 2560 in dy = 10 cm steps)
        Y = np.arange(0,256*10,10.)

        #Creation of the array to store the optical depth values
        opt_depth = np.zeros((self.nx,self.ny,self.nz))

        #Calculating the optical depth with the kappa integral
        for ix in range(self.nx):
            for iz in range(self.nz):
                for iy in range(self.ny):
                    #The first value of the top 
                    if iy == 0:
                        opt_depth[ix,self.ny-1,iz] = np.log10(kappa_cube[ix,self.ny-1:,iz])
                    else:
                        print(len(kappa_cube[ix,self.ny-1-iy:,iz]))
                        print(len(Y[self.ny-1-iy:]))
                        a = simps(kappa_cube[ix,self.ny-1-iy:,iz], x = None, dx = 10)
                        # Base 10 logarithm of the original optical depth
                        opt_depth[ix,self.ny-1-iy,iz] = np.log10(a)+3 #We are summing value of three here to obtain the 
                                                                        #ideal magnitudes, however we need to solve how to 
                                                                        #obtain those magnitude values without doing it by hand
        np.save(self.ptm+f"optical_depth_{filename}.npy", opt_depth)


if __name__ == "__main__":
    main()