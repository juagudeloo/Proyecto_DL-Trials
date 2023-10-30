import numpy as np
import train_generate.model_prof_tools as mpt
import pandas as pd
from scipy.interpolate import interp1d
from opt_depth_params import opt_len_val


"""
The class DataClass is used for the charging and classification of the polarimetric and MHD
data for the training and testing phases.
"""

class ChargeData():
    def __init__(self, ptm):
        """
        ptm: path to the data.
        """
        self.ptm = ptm

        #Number of wavelength points - its from 6300 amstroengs in steps of 10 amstroengs.
        self.nlam = 300

        #Dimensions of the MHD MuRAM simulation domain.
        #ny corresponds to the vertical dimension.
        self.nx, self.ny, self.nz = 480, 256, 480       
    def charge_atm_params(self, filename):
        """
        filename: number of the time step to access and charge the corresponding file data.
        """

        self.filename = filename
        #Arrays for saving the charged data for each filename
        # mtpr: temperature, mrho: density, mvyy: vertical component momentum (with the density), 
        # mbyy: vertical component magnetic field
        mtpr = []
        mrho = []
        mvyy = []
        mbyy = []

        """
        ======================================================================================
        Data charging
        ======================================================================================
        """
        
        #TEMPERATURE
        print("Charging temperature data...")
        #From charging the EOS data, we obtain the temperature information by taking the 0 index
        #in the 0 axis.
        mtpr = np.memmap.reshape(np.memmap(self.ptm+"eos."+self.filename,dtype=np.float32), 
                                (2,self.nx,self.ny,self.nz), order="A")[0,:,:,:]
        print("Temperature done!")
        
        #MAGNETIC FIELD VERTICAL COMPONENT
        print("Charging magn. field vertical component...")
        #Conversion coefficient for mbyy to be converted in cgs units.
        coef = np.sqrt(4.0*np.pi)
        mbyy =  coef*np.memmap(self.ptm+"result_6."+self.filename,dtype=np.float32)
        print("Magn. field verical component done!")

        #DENSITY
        print("Charging density...")
        mrho = np.memmap(self.ptm+"result_0."+self.filename,dtype=np.float32)
        print("Density done!")

        #VELOCITY VERTICAL COMPONENT
        #We divide the charged data by the density to obtain the actual velocity from the momentum.
        print("Charging velocity vertical component...")
        mvyy = np.memmap(self.ptm+"result_2."+self.filename,dtype=np.float32) / mrho
        print("Velocity vertical component done!")

        """
        ======================================================================================
        Applying logarithm to the data to work with magnitude order
        ======================================================================================
        """
        print("Finding logarithm for temperature, density, velocity and magnetic field")
        mtpr = np.log10(mtpr)
        mbyy = np.log10(mbyy)
        mrho = np.log10(mrho)
        mvyy = np.log10(mvyy)
        print("Logarithm done!")

        """
        ======================================================================================
        Array of the atm parameters (temperature, magnetic field, density and velocity)
        ======================================================================================
        """

        #Reshaping from nx,nz to nx*nz for the training. nx*nz will determine the number of data points
        #we have for the training, ny will be the number of points for the column of data and 4 is the 
        #number of "channels" given by the number of magnitudes.

        self.atm_params = [mtpr, mbyy, mrho, mvyy]
        return np.memmap.reshape(self.atm_params, (self.nx, self.nz, self.ny, 4))
    def remmap_opt_depth(self, filename):
        """
        Function for remmaping the atm params from height points to optical depth points in the vertical
        direction

        filename: number of the time step to access and charge the corresponding file data.
        """ 
        self.filename = filename
        self.charge_atm_params(filename)
        opt_depth = np.load(self.ptm+"optical_depth_"+filename+".npy")
        mags_names = ["By_opt", "Vy_opt", "log_rho_opt", "T_opt"]
        opt_mags_interp = {}

        self.opt_len = opt_len_val() #number of optical depth points
        tau = np.linspace(-3, 1, self.opt_len)
        opt_mags = np.zeros((self.nx, self.nz, self.opt_len, 4))#mbyy, #mvyy, #log(mrho), #mtpr
        ix, iz = 200,200
        for ix in range(self.nx):
            for iz in range(self.nz):
                for i in range(4):
                    opt_mags_interp[mags_names[i]] = interp1d(opt_depth[ix,self.lb:self.tb,iz], self.atm_params[ix,iz,:,i])
                    opt_mags[ix,iz,:,i] = opt_mags_interp[mags_names[i]](tau)

        self.atm_params = opt_mags
    def charge_stokes_params(self, filename):
        """
        Function for remmaping the stokes params calculated with the radiative transfer code NICOLE.

        filename: number of the time step to access and charge the corresponding file data.
        """ 
        self.filename = filename
        self.profs = np.load(self.ptm+filename+"_prof.npy")
        N_profs = 4
        self.profs = np.memmap.reshape(self.profs,(self.nx, self.nz, self.nlam, N_profs))
        return self.profs

class SplitData1D(ChargeData):
    def __init__(self, ptm):
        """
        ptm: path to the data.
        """
        ChargeData.__init__(self, ptm)
        
    def split_data(self, filename, TR_S):
        """
        Function for spliting the data in training and testing set.

        filename: number of the time step to access and charge the corresponding file data.
        TR_S: relative ratio from the whole data selected as the training set.
        """ 

        # Charging the spectropolarimetric and atmosphere magnitudes.
        self.charge_stokes_params(filename)
        self.charge_atm_params(filename)
        """
        ============================================================================
        Splitting the stokes parameters data. TR_S of the whole data is selected as 
        training set.
        ============================================================================
        """
        #Calculation of the threshold for the intergranular and granular zones
        


