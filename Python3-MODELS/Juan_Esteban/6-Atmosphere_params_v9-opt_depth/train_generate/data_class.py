import numpy as np
import train_generate.model_prof_tools as mpt
import pandas as pd
from scipy.interpolate import interp1d
from train_generate.boundaries import low_boundary, top_boundary


#This is the scaling function
def scaling(array, scaler_file_name, create_scaler=None):
    array1 = np.memmap.reshape(array,(-1,1))
    if create_scaler == True:
        scaler_pair = [np.ndarray.max(array1),
                        np.ndarray.min(array1)]
        np.save(f"{scaler_file_name}.npy", scaler_pair)
    elif create_scaler == False: 
        scaler = pd.read_csv("train_generate/scaler/scaler_pairs.csv")
        minimum = scaler[scaler_file_name].loc[0]
        maximum = scaler[scaler_file_name].loc[1]
        array1 = (array-minimum)/(maximum-minimum)
        array1 = np.ravel(array1)
        return array1
    else: raise ValueError("Inserted a non boolean value")

def inverse_scaling(array, scaler_file_name):
    scaler = pd.read_csv("train_generate/scaler/scaler_pairs.csv")
    minimum = scaler[scaler_file_name].loc[0]
    maximum = scaler[scaler_file_name].loc[1]
    array1 = array*(maximum-minimum)+minimum
    array1 = np.ravel(array1)
    return array1

#Here we import the class of nn_model.py to add to it the charging of the data, 
#the scaling of the input and the de-scaling of the output
class DataClass():
    def __init__(self, ptm, nx = 480, ny = 256, nz = 480, low_boundary = low_boundary(), top_boundary = top_boundary(), create_scaler = False, 
    light_type = "Intensity"): 
        """
        top_boundary -> indicates from where to take the data for training from top.
        low_boundary -> indicates from where to take the data for training from bottom.
        light_type options:
        "Intensity" -> The model predicts intensity.
        "Stokes params" -> The model predicts the Stokes parameters.
        create_scaler -> Set True by default. It determines wheter to create a scaler object or take an already created one.
        """

        self.nlam = 300 #this parameter is useful when managing the Stokes parameters #wavelenght interval - its from 6300 amstroengs in steps of 10 amstroengs
        self.ptm = ptm
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.tb = top_boundary
        self.lb = low_boundary
        self.create_scaler = create_scaler
        self.light_type = light_type
        print("Starting the charging process!")
    def charge_atm_params(self, filename, scale = True):
        #path and filename specifications
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
        print(f"reading EOS {self.filename}")
        #Charging temperature data
        self.mtpr = np.memmap(self.ptm+"eos."+self.filename,dtype=np.float32)
        self.mtpr = np.memmap.reshape(self.mtpr, (2,self.nx,self.ny,self.nz), order="A")
        n_eos = 0
        self.mtpr = self.mtpr[n_eos,:,:,:] 
        # n_eos -> 0: temperature ; 1: pressure
        if scale == True:
            print("scaling...")
            if self.create_scaler == True:
                scaling(self.mtpr, "mtpr", self.create_scaler)
            else:
                self.mtpr = scaling(self.mtpr, "mtpr", self.create_scaler)
        self.mtpr = ravel_xz(self.mtpr)[:,self.lb:self.tb] #we just want the upper half of the parameter values
        print(f"EOS done {self.filename}")
        print('\n')
        
        

        #Charging line of sight magnetic field components
        print (f"reading byy {self.filename}")
        self.mbyy = np.memmap(self.ptm+"result_6."+self.filename,dtype=np.float32)
        coef = np.sqrt(4.0*np.pi) #cgs units conversion300
        self.mbyy=self.mbyy*coef
        if scale == True:
            if self.create_scaler == True:
                scaling(self.mbyy, "mbyy", self.create_scaler)
            else:
                self.mbyy = scaling(self.mbyy, "mbyy", self.create_scaler)
        self.mbyy = ravel_xz(self.mbyy)[:,self.lb:self.tb] #we just want the upper half of the parameter values
        print(f"byy done {self.filename}")
        print('\n')

        #Charging density values
        print(f"reading rho and mvyy (dividing mvyy/mrho to obtain vyy) {self.filename}")
        self.mrho = np.memmap(self.ptm+"result_0."+self.filename,dtype=np.float32)
        self.mvyy = np.memmap(self.ptm+"result_2."+self.filename,dtype=np.float32)
        self.mvyy = self.mvyy/self.mrho #obtaining the velocity from the momentum values
        
        self.mrho = np.log10(self.mrho)
        if scale == True:
            if self.create_scaler == True:
                scaling(self.mrho, "mrho", self.create_scaler)
            else:
                self.mrho = scaling(self.mrho, "mrho", self.create_scaler)
            if self.create_scaler == True:
                scaling(self.mvyy, "mvyy", self.create_scaler)
            else:
                self.mvyy = scaling(self.mvyy, "mvyy", self.create_scaler)
        self.mrho = ravel_xz(self.mrho)[:,self.lb:self.tb]
        self.mvyy = ravel_xz(self.mvyy)[:,self.lb:self.tb] #we just want the upper half of the parameter values

        print(f"rho and vyy done {self.filename}")
        print('\n')

        #Organizing the input data
        self.atm_params = [self.mbyy, self.mvyy, self.mrho, self.mtpr]
        self.atm_params = np.array(self.atm_params)
        
        #because the data is ravel, the atm_params has originally the shape (4, nx*nz, 256-lb)
        self.atm_params = np.moveaxis(self.atm_params,0,1) #(nx*nz, 4, 256-lb)
        self.atm_params = np.moveaxis(self.atm_params,1,2) #(nx*nz, 256-lb, 4)
        self.atm_params = np.memmap.reshape(self.atm_params, (self.nx, self.nz, (self.tb-self.lb), 4))
        return np.memmap.reshape(self.atm_params, (self.nx, self.nz, (self.tb-self.lb), 4))
    def remmap_opt_depth(self, filename):
        self.filename = filename
        self.charge_atm_params(filename)
        opt_depth = np.load(self.ptm+"optical_depth_"+filename+".npy")
        mags_names = ["By_opt", "Vy_opt", "log_rho_opt", "T_opt"]
        opt_mags_interp = {}
        N = 50
        tau = np.linspace(-5, 0.5, N)
        opt_mags = np.zeros((self.nx, self.ny, N, 4))#mbyy, #mvyy, #log(mrho), #mtpr
        ix, iz = 200,200
        for ix in range(self.nx):
            for iz in range(self.nz):
                for i in range(4):
                    print("opt_depth length:", opt_depth[ix,:,iz].shape)
                    print("atm_paramas length:", self.atm_params[ix,iz,:,i.shape])
                    opt_mags_interp[mags_names[i]] = interp1d(opt_depth[ix,:,iz], self.atm_params[ix,iz,:,i])
                    opt_mags[ix,iz,:,i] = opt_mags_interp[mags_names[i]](tau)
        print(opt_mags.shape)
        
    def charge_intensity(self,filename,scale = True):
        self.filename = filename
        self.iout = []
        print(f"reading IOUT {self.filename}")
        self.iout = np.memmap(self.ptm+"iout."+self.filename,dtype=np.float32)
        if scale == True:
            if self.create_scaler == True:
                scaling(self.iout, "iout", self.create_scaler) #scaled intensity
            else:
                print("scaling...")
                self.iout = scaling(self.iout, "iout", self.create_scaler) #scaled intensity
        self.iout = np.memmap.reshape(self.iout,(self.nx, self.nz))
        print(f"IOUT done {self.filename}")   
        print('\n') 
        return self.iout
    def charge_stokes_params(self, filename,  file_type = "nicole", scale = True):
        import struct
        import re
        import sys
        global idl, irec, f # Save values between calls
        self.filename = filename
        [int4f,intf,flf]=mpt.check_types()
        self.stk_filename = self.filename+"_0000_0000.prof"
        self.profs = [] #It's for the reshaped data - better for visualization.
        self.profs_ravel = [] #its for the ravel data to make the splitting easier.
        #Charging the stokes profiles for the specific file
        print(f"reading Stokes params {self.stk_filename}")
        N_profs = 4
        for ix in range(self.nx):
            for iy in range(self.nz):
                p_prof = mpt.read_prof(self.ptm+self.stk_filename, file_type,  self.nx, self.nz, self.nlam, iy, ix)
                p_prof = np.memmap.reshape(np.array(p_prof), (self.nlam, N_profs))
                ##############################################################################
                #self.profs_ravel is going to safe all the data in a one dimensional array where
                #the dimensional indexes are disposed as ix*self.nz+iy.
                ##############################################################################
                self.profs.append(p_prof)  
        print("scaling...")
        self.profs = np.array(self.profs) #this step is done so that the array has the same shape as the ouputs referring to the four type of data it has
        #We scale all the stokes parameters under the same scaler because all of them belong to the same whole Intensity physical phenomenon
        if scale == True:
            if self.create_scaler == True:
                scaling(self.profs, "stokes", self.create_scaler)
            else:
                self.profs = scaling(self.profs, "stokes", self.create_scaler)
        else:
            None
        #for i in range(N_profs):
        #    self.profs[:,i,:] = np.memmap.reshape(self.profs[:,i,:],(self.nx*self.nz, self.nlam))
        #Here we are flattening the whole values of the four stokes parameters into a single axis to set them as a one array ouput to the nn model
        self.profs = np.memmap.reshape(self.profs,(self.nx, self.nz, self.nlam, N_profs))
        print(f"Stokes params done! {self.filename}")
        return self.profs
    def split_data_atm_output(self, filename, light_type, TR_S):
        """
        Splits the data into a test set and a training set.
        It is a hand made splitting.
        TR_S: relative ratio of the whole data selected to the training set.
        """
        self.light_type = light_type
        #Arrays of input and output training and testing sets
        #Delimiting the training and testing sets
        if self.light_type == "Intensity":
            self.charge_intensity(filename)
            # Creating the masks for the granular and intergranular zones
            threshold = (np.max(self.iout)-np.min(self.iout))/2.5 + np.min(self.iout)
            intergran_mask = np.ma.masked_where(self.iout > threshold, self.iout).mask
            gran_mask = np.ma.masked_where(self.iout <= threshold, self.iout).mask
            len_intergran = len(np.ma.masked_where(self.iout > threshold, self.iout).compressed())
            len_gran = len(np.ma.masked_where(self.iout <= threshold, self.iout).compressed())
            # Applying the masks over the intensity data
            iout_intergran = np.ma.array(self.iout, mask = intergran_mask).compressed()
            iout_gran = np.ma.array(self.iout, mask = gran_mask).compressed()
            # Creating the array of random indexes for the granular zones, to obtain the same total of data than the intergranular zones.
            index_select  = []
            np.random.seed(50)
            if len_intergran < len_gran:
                index_select = np.random.choice(range(len_gran), size = (len_intergran,), replace = False)
            else:
                raise ValueError("Intergranular points should always be less than granular points")
            # Determination of the training and testing set inputs.
            idx = np.arange(len_intergran) 
            np.random.shuffle(idx) #shufling this indexes to obtain a random training subset selection of the original set of data.
            TR_delim = int(len_intergran*TR_S)

            self.tr_input = np.concatenate((iout_intergran[idx[:TR_delim]], iout_gran[index_select][idx[:TR_delim]]))
            self.te_input = np.concatenate((iout_intergran[idx[TR_delim:]], iout_gran[index_select][idx[TR_delim:]]))
            self.in_ls = (1,) #input shape for the neural network

        if self.light_type == "Stokes params":
            self.charge_stokes_params(filename)
            threshold = (np.max(self.profs[:,:,0,0])-np.min(self.profs[:,:,0,0]))/2.5 + np.min(self.profs[:,:,0,0])
            intergran_mask = np.ma.masked_where(self.profs[:,:,0,0] > threshold, self.profs[:,:,0,0]).mask
            gran_mask = np.ma.masked_where(self.profs[:,:,0,0] <= threshold, self.profs[:,:,0,0]).mask
            len_intergran = len(np.ma.masked_where(self.profs[:,:,0,0] > threshold, self.profs[:,:,0,0]).compressed())
            len_gran = len(np.ma.masked_where(self.profs[:,:,0,0] <= threshold, self.profs[:,:,0,0]).compressed())
            # Stokes profiles

            profile_intergran = []
            profile_gran = []
            p_in = []
            p_gran = []

            for j in range(4):
                p_in = []
                p_gran = []
                for i in range(300):
                    p_in.append(np.ma.array(self.profs[:,:,i,j], mask = intergran_mask).compressed())
                    p_gran.append(np.ma.array(self.profs[:,:,i,j], mask = gran_mask).compressed())
                profile_intergran.append(p_in)
                profile_gran.append(p_gran)

            profile_intergran = np.array(profile_intergran)
            profile_gran = np.array(profile_gran)
            for i in range(2):
                profile_intergran = np.moveaxis(profile_intergran,0,2-i)
                profile_gran = np.moveaxis(profile_gran,0,2-i)
            # Creating the array of random indexes for the granular zones, to obtain the same total of data than the intergranular zones.
            index_select  = []
            np.random.seed(50)
            if len_intergran < len_gran:
                index_select = np.random.choice(range(len_gran), size = (len_intergran,), replace = False)
            else:
                raise ValueError("Intergranular points should always be less than granular points")
            # Determination of the training and testing set inputs.
            idx = np.arange(len_intergran) 
            np.random.shuffle(idx) #shufling this indexes to obtain a random training subset selection of the original set of data.
            TR_delim = int(len_intergran*TR_S)

            self.tr_input = np.concatenate((profile_intergran[idx[:TR_delim],:,:], profile_gran[index_select,:,:][idx[:TR_delim],:,:]))
            self.te_input = np.concatenate((profile_intergran[idx[TR_delim:],:,:], profile_gran[index_select,:,:][idx[TR_delim:],:,:]))
            self.in_ls = np.shape(profile_intergran[0,:,:]) #input shape for the neural network
        print(f"in_ls = {self.in_ls}")


        # Atmosphere params
        self.charge_atm_params(filename)
        atm_intergran = []
        atm_gran = []
        a_in = []
        a_gran = []

        for j in range(4):
            a_in = []
            a_gran = []
            for i in range(self.tb-self.lb):
                a_in.append(np.ma.array(self.atm_params[:,:,i,j], mask = intergran_mask).compressed())
                a_gran.append(np.ma.array(self.atm_params[:,:,i,j], mask = gran_mask).compressed())
            atm_intergran.append(a_in)
            atm_gran.append(a_gran)
        
        atm_intergran = np.array(atm_intergran)
        atm_gran = np.array(atm_gran)

        for i in range(2):
            atm_intergran = np.moveaxis(atm_intergran,0,2-i)
            atm_gran = np.moveaxis(atm_gran,0,2-i)
        
        self.tr_output = np.concatenate((atm_intergran[idx[:TR_delim],:,:], atm_gran[index_select,:,:][idx[:TR_delim],:,:]))
        self.te_output = np.concatenate((atm_intergran[idx[TR_delim:],:,:], atm_gran[index_select,:,:][idx[TR_delim:],:,:]))

        print(self.tr_input.shape) 
        print(self.te_input.shape) 
        print(self.tr_output.shape)
        print(self.te_output.shape)

        self.tr_output = np.memmap.reshape(self.tr_output, (np.shape(self.tr_output)[0], np.shape(self.tr_output)[1]*np.shape(self.tr_output)[2]), order = "A")
        self.te_output = np.memmap.reshape(self.te_output, (np.shape(self.te_output)[0], np.shape(self.te_output)[1]*np.shape(self.te_output)[2]), order = "A")


        print(self.tr_input.shape) 
        print(self.te_input.shape) 
        print(self.tr_output.shape)
        print(self.te_output.shape)
        return self.tr_input, self.tr_output, self.te_input, self.te_output
    def split_data_light_output(self, filename, light_type, TR_S):
        """
        Splits the data into a test set and a training set.
        It is a hand made splitting.
        TR_S: relative ratio of the whole data selected to the training set.
        """
        #Light information

        self.light_type = light_type
        #Arrays of input and output training and testing sets
        #Delimiting the training and testing sets
        if self.light_type == "Intensity":
            self.charge_intensity(filename)
            # Creating the masks for the granular and intergranular zones
            threshold = (np.max(self.iout)-np.min(self.iout))/2.5 + np.min(self.iout)
            intergran_mask = np.ma.masked_where(self.iout > threshold, self.iout).mask
            gran_mask = np.ma.masked_where(self.iout <= threshold, self.iout).mask
            len_intergran = len(np.ma.masked_where(self.iout > threshold, self.iout).compressed())
            len_gran = len(np.ma.masked_where(self.iout <= threshold, self.iout).compressed())
            # Applying the masks over the intensity data
            iout_intergran = np.ma.array(self.iout, mask = intergran_mask).compressed()
            iout_gran = np.ma.array(self.iout, mask = gran_mask).compressed()
            # Creating the array of random indexes for the granular zones, to obtain the same total of data than the intergranular zones.
            index_select  = []
            np.random.seed(50)
            if len_intergran < len_gran:
                index_select = np.random.choice(range(len_gran), size = (len_intergran,), replace = False)
            else:
                raise ValueError("Intergranular points should always be less than granular points")
            # Determination of the training and testing set outputs.
            idx = np.arange(len_intergran) 
            np.random.shuffle(idx) #shufling this indexes to obtain a random training subset selection of the original set of data.
            TR_delim = int(len_intergran*TR_S)

            self.tr_output = np.concatenate((iout_intergran[idx[:TR_delim]], iout_gran[index_select][idx[:TR_delim]]))
            self.te_output = np.concatenate((iout_intergran[idx[TR_delim:]], iout_gran[index_select][idx[TR_delim:]]))

        if self.light_type == "Stokes params":
            self.charge_stokes_params(filename)
            threshold = (np.max(self.profs[:,:,0,0])-np.min(self.profs[:,:,0,0]))/2.5 + np.min(self.profs[:,:,0,0])
            intergran_mask = np.ma.masked_where(self.profs[:,:,0,0] > threshold, self.profs[:,:,0,0]).mask
            gran_mask = np.ma.masked_where(self.profs[:,:,0,0] <= threshold, self.profs[:,:,0,0]).mask
            len_intergran = len(np.ma.masked_where(self.profs[:,:,0,0] > threshold, self.profs[:,:,0,0]).compressed())
            len_gran = len(np.ma.masked_where(self.profs[:,:,0,0] <= threshold, self.profs[:,:,0,0]).compressed())
            self.charge_stokes_params(filename)
            # Stokes profiles

            profile_intergran = []
            profile_gran = []
            p_in = []
            p_gran = []

            for j in range(4):
                p_in = []
                p_gran = []
                for i in range(300):
                    p_in.append(np.ma.array(self.profs[:,:,i,j], mask = intergran_mask).compressed())
                    p_gran.append(np.ma.array(self.profs[:,:,i,j], mask = gran_mask).compressed())
                profile_intergran.append(p_in)
                profile_gran.append(p_gran)

            profile_intergran = np.array(profile_intergran)
            profile_gran = np.array(profile_gran)
            for i in range(2):
                profile_intergran = np.moveaxis(profile_intergran,0,2-i)
                profile_gran = np.moveaxis(profile_gran,0,2-i)
            # Creating the array of random indexes for the granular zones, to obtain the same total of data than the intergranular zones.
            index_select  = []
            np.random.seed(50)
            if len_intergran < len_gran:
                index_select = np.random.choice(range(len_gran), size = (len_intergran,), replace = False)
            else:
                raise ValueError("Intergranular points should always be less than granular points")
            # Determination of the training and testing set outputs.
            idx = np.arange(len_intergran) 
            np.random.shuffle(idx) #shufling this indexes to obtain a random training subset selection of the original set of data.
            TR_delim = int(len_intergran*TR_S)

            self.tr_output = np.concatenate((profile_intergran[idx[:TR_delim],:,:], profile_gran[index_select,:,:][idx[:TR_delim],:,:]))
            self.te_output = np.concatenate((profile_intergran[idx[TR_delim:],:,:], profile_gran[index_select,:,:][idx[TR_delim:],:,:]))

            self.tr_output = np.memmap.reshape(self.tr_output, (np.shape(self.tr_output)[0], np.shape(self.tr_output)[1]*np.shape(self.tr_output)[2]), order = "A")
            self.te_output = np.memmap.reshape(self.te_output, (np.shape(self.te_output)[0], np.shape(self.te_output)[1]*np.shape(self.te_output)[2]), order = "A")
            
        # Atmosphere params
        self.charge_atm_params(filename)
        atm_intergran = []
        atm_gran = []
        a_in = []
        a_gran = []

        for j in range(4):
            a_in = []
            a_gran = []
            for i in range(self.tb-self.lb):
                a_in.append(np.ma.array(self.atm_params[:,:,i,j], mask = intergran_mask).compressed())
                a_gran.append(np.ma.array(self.atm_params[:,:,i,j], mask = gran_mask).compressed())
            atm_intergran.append(a_in)
            atm_gran.append(a_gran)
        
        atm_intergran = np.array(atm_intergran)
        atm_gran = np.array(atm_gran)
        
        for i in range(2):
            atm_intergran = np.moveaxis(atm_intergran,0,2-i)
            atm_gran = np.moveaxis(atm_gran,0,2-i)
            
        self.tr_input = np.concatenate((atm_intergran[idx[:TR_delim],:,:], atm_gran[index_select,:,:][idx[:TR_delim],:,:]))
        self.te_input = np.concatenate((atm_intergran[idx[TR_delim:],:,:], atm_gran[index_select,:,:][idx[TR_delim:],:,:]))
        self.in_ls = np.shape(atm_intergran[0,:,:])
        print(f"in_ls = {self.in_ls}")

        

        return self.tr_input, self.tr_output, self.te_input, self.te_output

        

	

        








