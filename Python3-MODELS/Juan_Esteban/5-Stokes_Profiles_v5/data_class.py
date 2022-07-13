from cmath import inf, nan
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import model_prof_tools as mpt
import testing_functions as tef

#This is the scaling function
def scaling(array):
    scaler = MinMaxScaler()
    array1 = array.reshape(-1,1)
    scaler.fit(array1)
    array1 = scaler.transform(array1)
    array1 = np.ravel(array1)
    return array1

#Here we import the class of nn_model.py to add to it the charging of the data, 
#the scaling of the input and the de-scaling of the output
class Data_class():
    def __init__(self, nx = 480, ny = 256, nz = 480): 
        """
        output_type options:
        "Intensity" -> The model predicts intensity.
        "Stokes params" -> The model predicts the Stokes parameters.
        """
        #size of the cubes of the data
        self.nx = nx
        self.ny = ny
        self.nz = nz
        print("Starting the charging process!")
    def charge_inputs(self, filename, ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"):
        #path and filename specifications
        self.ptm = ptm
        self.filename = filename
        #Arrays for saving the charged data for each filename
        self.mtpr = []
        self.mrho = []
        self.mvyy = []
        self.mbyy = []
        #Arrays for saving the charged data for each filename and raveled
        self.mvyy_ravel = []
        self.mbyy_ravel = []
        self.mtpr_ravel = []
        self.mrho_ravel = []
        coef = np.sqrt(4.0*np.pi) #for converting data to cgs units
        #Function for raveling the nx and nz coordinates after the processing
        def ravel_xz(array):
                array_ravel = array.reshape(self.nx, self.ny, self.nz)
                array_ravel = np.moveaxis(array_ravel,1,2)
                array_ravel = array_ravel.reshape(self.nx*self.nz, self.ny) 
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
        print(f"reading EOS {self.filename}")
        #Charging temperature data
        self.mtpr = np.memmap(self.ptm+"eos."+self.filename,dtype=np.float32)
        self.mtpr = np.reshape(self.mtpr, (2,self.nx,self.ny,self.nz), order="A")
        n_eos = 0
        self.mtpr = self.mtpr[n_eos,:,:,:] 
        # n_eos -> 0: temperature ; 1: pressure
        self.mtpr = scaling(self.mtpr)
        self.mtpr = ravel_xz(self.mtpr)
        print(f"EOS done {self.filename}")
        print('\n')
        
        

        #Charging line of sight magnetic field components
        print (f"reading byy {self.filename}")
        self.mbyy = np.memmap(self.ptm+"result_6."+self.filename,dtype=np.float32)
        coef = np.sqrt(4.0*np.pi) #cgs units conversion
        self.mbyy=self.mbyy*coef
        self.mbyy = scaling(self.mbyy)
        self.mbyy = ravel_xz(self.mbyy)
        print(f"byy done {self.filename}")
        print('\n')

        #Charging density values
        print(f"reading rho and mvyy (dividing mvyy/mrho to obtain vyy) {self.filename}")
        self.mrho = np.memmap(self.ptm+"result_0."+self.filename,dtype=np.float32)
        self.mvyy = np.memmap(self.ptm+"result_2."+self.filename,dtype=np.float32)
        self.mvyy = self.mvyy/self.mrho #obtaining the velocity from the momentum values
        
        self.mrho = scaling(self.mrho)
        self.mrho = ravel_xz(self.mrho)
        self.mvyy = scaling(self.mvyy)
        self.mvyy = ravel_xz(self.mvyy)
        print(f"rho and vyy done {self.filename}")
        print('\n')
        
        #Organizing the input data
        self.input_values = [self.mbyy, self.mvyy, self.mrho, self.mtpr]
        self.input_values = np.array(self.input_values)
        self.input_values = np.moveaxis(self.input_values,0,1)
        return self.input_values
    def charge_intensity(self,filename, ptm = "/mnt/scratch/juagudeloo/Total_MURAM_data/"):
        self.ptm = ptm
        self.filename = filename
        self.iout = []
        print(f"reading IOUT {self.filename}")
        self.iout = np.memmap(self.ptm+"iout."+self.filename,dtype=np.float32)
        print("scaling...")
        self.iout = scaling(self.iout) #scaled intensity
        print(np.shape(self.iout))
        print(f"IOUT done {self.filename}")   
        print('\n') 
        return self.iout
    def charge_stokes_params(self, filename, stk_ptm = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/",  file_type = "nicole"):
        self.stk_ptm = stk_ptm
        self.stk_filename = filename+"_0000_0000.prof"
        self.nlam = 300 #wavelenght interval - its from 6300 amstroengs-
        self.profs = [] #It's for the reshaped data - better for visualization.
        self.profs_ravel = [] #its for the ravel data to make the splitting easier.
        #Charging the stokes profiles for the specific file
        print(f"reading Stokes params {self.stk_filename}")
        N_profs = 4
        for ix in range(self.nx):
            for iy in range(self.nz):
                p_prof = mpt.read_prof(self.stk_ptm+self.stk_filename, file_type,  self.nx, self.nz, self.nlam, iy, ix)
                p_prof = np.reshape(p_prof, (self.nlam, N_profs))
                ##############################################################################
                #self.profs_ravel is going to safe all the data in a one dimensional array where
                #the dimensional indexes are disposed as ix*self.nz+iy.
                ##############################################################################
                self.profs.append(p_prof) 
        print("scaling...")
        self.profs = np.array(self.profs) 
        self.profs = np.moveaxis(self.profs,1,2) #this step is done so that the array has the same shape as the ouputs referring to the four type of data it has
        for i in range(N_profs):
            self.profs[:,i,:] = scaling(self.profs[:,i,:].reshape(self.nx*self.nz, self.nlam))
        #Here we are flattening the whole values of the four stokes parameters into a single axis to set them as a one array ouput to the nn model
        self.profs = self.profs.reshape(self.nx*self.nz,N_profs*self.nlam) 
        print(f"Stokes params done! {self.filename}")
        return self.profs
    def split_data(self, filename, output_type, TR_S):
        """
        Splits the data into a test set and a training set.
        It is a hand made splitting.
        TR_S: relative ratio of the whole data selected to the training set.
        """
        self.output_type = output_type
        #Arrays of input and output training and testing sets
        self.tr_input = []
        self.te_input = []
        self.tr_output = []
        self.te_output = []
        #Delimiting the training and testing sets
        self.charge_inputs(filename)
        n_data = self.input_values[:,0,0].size
        idx = np.arange(n_data) 
        np.random.shuffle(idx) #shufling this indexes to obtain a random training subset selection of the original set of data.
        TR_delim = int(n_data*TR_S)
        self.tr_input = self.input_values[idx[:TR_delim]]
        self.te_input = self.input_values[idx[TR_delim:]]
        if self.output_type == "Intensity":
            self.charge_intensity(filename)
            self.tr_output = self.iout[idx[:TR_delim]]
            self.te_output = self.iout[idx[TR_delim:]]
        if self.output_type == "Stokes params":
            self.charge_stokes_params(filename)
            self.tr_output = self.profs[idx[:TR_delim]]
            self.te_output = self.profs[idx[TR_delim:]]
        return self.tr_input, self.tr_output, self.te_input, self.te_output




        

	

        








