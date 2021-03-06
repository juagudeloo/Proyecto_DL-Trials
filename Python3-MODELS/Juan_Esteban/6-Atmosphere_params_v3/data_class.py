from cmath import inf, nan
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import model_prof_tools as mpt

#This is the scaling function
def scaling(array):
    scaler = MinMaxScaler()
    array1 = np.memmap.reshape(array,(-1,1))
    scaler.fit(array1)
    array1 = scaler.transform(array1)
    array1 = np.ravel(array1)
    return array1

#Here we import the class of nn_model.py to add to it the charging of the data, 
#the scaling of the input and the de-scaling of the output
class Data_class_indiv():
    def __init__(self, physical_magnitude, nx = 480, ny = 256, nz = 480): 
        #size of the cubes of the data
        self.phys_mag = physical_magnitude
        self.nx = nx
        self.ny = ny
        self.nz = nz
        print("Starting the charging process!")
    def charge_magnitude(self, filename, ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"):
        #path and filename specifications
        self.ptm = ptm
        self.filename = filename
        #Arrays for saving the charged data for each filename
        self.magnitude = []
        #Function for raveling the nx and nz coordinates after the processing
        def ravel_xz(array):
                array_ravel = np.memmap.reshape(array,(self.nx, self.ny, self.nz))
                array_ravel = np.moveaxis(array_ravel,1,2)
                array_ravel = np.memmap.reshape(array_ravel,(self.nx*self.nz, self.ny))
                return array_ravel
        ################################
        # Charging the data into the code - every data is converted into a cube 
        # of data so that it has the form of the dominium of the simulation
        #
        # The lines of the code for mvxx, mvyy, mbxx and mbyy are commented beacuse
        # we are not interested in ploting this magnitudes for the image analysis 
        # we are going to make
        #
        # The temperature is obtained from the data file related to the 
        # equation of state (EOS)
        ################################
        if self.phys_mag == "mbyy":
            #Charging line of sight magnetic field components
            print (f"reading byy {self.filename}")
            self.magnitude = np.memmap(self.ptm+"result_6."+self.filename,dtype=np.float32)
            coef = np.sqrt(4.0*np.pi) #cgs units conversion
            self.magnitude=self.magnitude*coef
            print(f"byy done {self.filename}")
        if self.phys_mag == "mvyy":
            print(f"reading rho and mvyy (dividing mvyy/mrho to obtain vyy) {self.filename}")
            mrho = np.memmap(self.ptm+"result_0."+self.filename,dtype=np.float32)
            mvyy = np.memmap(self.ptm+"result_2."+self.filename,dtype=np.float32)
            self.magnitude = mvyy/mrho #obtaining the velocity from the momentum values
            print(f"vyy done {self.filename}")
        if self.phys_mag == "mrho":
            print(f"reading rho {self.filename}")
            self.magnitude = np.memmap(self.ptm+"result_0."+self.filename,dtype=np.float32)
            print(f"rho done {self.filename}")
        if self.phys_mag == "mtpr":
            print(f"reading temperature {self.filename}")
            tpr = np.memmap(self.ptm+"eos."+self.filename,dtype=np.float32)
            tpr = np.memmap.reshape(self.mtpr, (2,self.nx,self.ny,self.nz), order="A")
            n_eos = 0
            self.magnitude = tpr[n_eos,:,:,:] 
            print(f"temperature done {self.filename}")
        print('\n')
        self.magnitude = scaling(self.magnitude)
        self.magnitude = ravel_xz(self.magnitude)
        return self.magnitude
    def charge_intensity(self,filename, ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"):
        self.ptm = ptm
        self.filename = filename
        self.iout = []
        print(f"reading IOUT {self.filename}")
        self.iout = np.memmap(self.ptm+"iout."+self.filename,dtype=np.float32)
        print("scaling...")
        self.iout = scaling(self.iout) #scaled intensity
        print(f"IOUT done {self.filename}")   
        print('\n') 
        return self.iout
    def charge_stokes_params(self, filename, stk_ptm = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/",  file_type = "nicole"):
        import struct
        import re
        import sys
        global idl, irec, f # Save values between calls
        self.filename = filename
        [int4f,intf,flf]=mpt.check_types()
        self.stk_ptm = stk_ptm
        self.stk_filename = self.filename+"_0000_0000.prof"
        self.nlam = 300 #wavelenght interval - its from 6300 amstroengs-
        self.profs = [] #It's for the reshaped data - better for visualization.
        self.profs_ravel = [] #its for the ravel data to make the splitting easier.
        #Charging the stokes profiles for the specific file
        print(f"reading Stokes params {self.stk_filename}")
        N_profs = 4
        for ix in range(self.nx):
            for iy in range(self.nz):
                p_prof = mpt.read_prof(self.stk_ptm+self.stk_filename, file_type,  self.nx, self.nz, self.nlam, iy, ix)
                p_prof = np.memmap.reshape(np.array(p_prof), (self.nlam, N_profs))
                ##############################################################################
                #self.profs_ravel is going to safe all the data in a one dimensional array where
                #the dimensional indexes are disposed as ix*self.nz+iy.
                ##############################################################################
                self.profs.append(p_prof)  
        print("scaling...")
        self.profs = np.array(self.profs) 
        self.profs = np.moveaxis(self.profs,1,2) #this step is done so that the array has the same shape as the ouputs referring to the four type of data it has
        for i in range(N_profs):
            self.profs[:,i,:] = np.memmap.reshape(scaling(self.profs[:,i,:]),(self.nx*self.nz, self.nlam))
        #Here we are flattening the whole values of the four stokes parameters into a single axis to set them as a one array ouput to the nn model
        self.profs = np.memmap.reshape(self.profs,(self.nx*self.nz,N_profs,self.nlam))
        print(f"Stokes params done! {self.filename}")
        return self.profs
    def split_data(self, filename, input_type, physical_magnitude, TR_S):
        """
        Splits the data into a test set and a training set.
        It is a hand made splitting.
        TR_S: relative ratio of the whole data selected to the training set.
        """
        self.output_type = input_type
        #Arrays of input and output training and testing sets
        self.tr_input = []
        self.te_input = []
        self.tr_output = []
        self.te_output = []
        #Delimiting the training and testing sets
        if self.output_type == "Intensity":
            self.charge_intensity(filename)
            n_data = self.iout[:,0].size
            idx = np.arange(n_data) 
            np.random.shuffle(idx) #shufling this indexes to obtain a random training subset selection of the original set of data.
            TR_delim = int(n_data*TR_S)
            self.tr_input = self.iout[idx[:TR_delim]]
            self.te_input = self.iout[idx[TR_delim:]]
        if self.output_type == "Stokes params":
            self.charge_stokes_params(filename)
            n_data = self.profs[:,0,0].size
            idx = np.arange(n_data) 
            np.random.shuffle(idx) #shufling this indexes to obtain a random training subset selection of the original set of data.
            TR_delim = int(n_data*TR_S)
            self.tr_input = self.profs[idx[:TR_delim]]
            self.te_input = self.profs[idx[TR_delim:]]
        self.charge_magnitude(filename)
        self.tr_output = self.magnitude[idx[:TR_delim]]
        self.te_output = self.magnitude[idx[TR_delim:]]
        return self.tr_input, self.tr_output, self.te_input, self.te_output




        

	

        








