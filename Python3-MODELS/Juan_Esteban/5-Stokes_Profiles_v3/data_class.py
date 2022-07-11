from cmath import inf, nan
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler
from nn_model import NN_Model #module in the same folder
import model_prof_tools as mpt

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
class Data_NN_model(NN_Model):
    def __init__(self, nx = 480, ny = 256, nz = 480):
        #size of the cubes of the data
        self.nx = nx
        self.ny = ny
        self.nz = nz
        print("Starting the charging process!")
    def charge_inputs(self, ptm, filename):
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
                array_ravel = np.moveaxis(array,1,2)
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
        if type(self.filename) == str: #if we just want to chek one file
            print(f"reading EOS {self.filename}")
            #Charging temperature data
            self.mtpr = np.memmap(self.ptm+"eos."+self.filename,dtype=np.float32)
            self.mtpr = np.reshape(self.mtpr, (2, self.nx,self.ny,self.nz), order="A")
            n_eos = 0
            self.mtpr = self.mtpr[n_eos,:,:,:] 
            # n_eos -> 0: temperature ; 1: pressure
            print(f"EOS done {self.filename}")
            print('\n')
            
            #Charging line of sight velocities
            print(f"reading vyy {self.filename}")
            self.mvyy = np.memmap(self.ptm+"result_2."+self.filename,dtype=np.float32)
            self.mvyy = np.reshape(self.mvyy,(self.nx,self.ny,self.nz),order="C")
            print(f"vyy done {self.filename}")
            print('\n')

            #Charging line of sight magnetic field components
            print (f"reading byy {self.filename}")
            self.mbyy = np.memmap(self.ptm+"result_6."+self.filename,dtype=np.float32)
            self.mbyy = np.reshape(self.mbyy,(self.nx,self.ny,self.nz),order="C")
            print(f"byy done {self.filename}")
            print('\n')

            #Charging density values
            print(f"reading rho {self.filename}")
            self.mrho = np.memmap(self.ptm+"result_0."+self.filename,dtype=np.float32)
            self.mrho = np.reshape(self.mrho, (self.nx,self.ny,self.nz), order="A")
            print("scaling...")
            if np.any(self.mrho == 0): #In case this condition happens, the logarithm will diverge, then we will ignore this kind of data by setting all the columns of all quantities related with this value to zero
                self.mrho = np.reshape(self.mrho, (self.nx,self.ny,self.nz), order="A")
                print(np.shape(self.mrho))
                print(type(self.mrho))
                print(f"rho done {self.filename[i]}")
                print('\n')

                #x,z coordinates where the density have zero values
                self.nx0 = np.argwhere(self.mrho == 0)[0][0]
                self.nz0 = np.argwhere(self.mrho == 0)[0][2]

                #I set this columns to ones beacuse due to the magnitude of each 
                #of this measurements, when scaling they are going to be almost negligible
                #but no zero, therefore there is not going to be any problem when dividing 
                #mvyy/mrho. It is also done so that this columns are discard from the 
                #fitting
                self.mrho[self.nx0,:,self.nz0] = np.ones(self.ny)
                self.mtpr[self.nx0,:,self.nz0] = np.ones(self.ny)
                self.mbyy[self.nx0,:,self.nz0] = np.ones(self.ny)
                self.mvyy[self.nx0,:,self.nz0] = np.ones(self.ny)
                #############################################################
                #Converting the data into cgs units (if I'm not wrong)
                #############################################################
                
            else:
                self.mvyy=self.mvyy/self.mrho
                self.mvyy = scaling(self.mvyy) #scaling
                self.mvyy = np.reshape(self.mvyy,(self.nx,self.ny,self.nz),order="C")

                self.mbyy = scaling(self.mbyy)
                self.mbyy = np.reshape(self.mbyy,(self.nx,self.ny,self.nz),order="C")

            #Scaling the charge data   
            print("scaling...")
            self.mvyy=self.mvyy/self.mrho
            self.mvyy = scaling(self.mvyy) #scaling
            self.mvyy = np.reshape(self.mvyy,(self.nx,self.ny,self.nz),order="C")

            self.mbyy = scaling(self.mbyy)
            self.mbyy = np.reshape(self.mbyy,(self.nx,self.ny,self.nz),order="C")

            self.mtpr = scaling(self.mtpr)
            self.mtpr = np.reshape(self.mtpr, (self.nx,self.ny,self.nz), order="A")
                
                
            self.mrho = np.log10(self.mrho) #I get the logarithm in base 10 out of the density values so that the big valued data does not damage the code
            self.mrho = scaling(self.mrho)
            self.mrho = np.reshape(self.mrho, (self.nx,self.ny,self.nz), order="A")

            #Saving ravel outputs for easier splitting
            self.mvyy_ravel = ravel_xz(self.mvyy)
            self.mbyy_ravel = ravel_xz(self.mbyy)
            self.mtpr_ravel = ravel_xz(self.mtpr)
            self.mrho_ravel = ravel_xz(self.mrho)

            

        else: #if filename is an array of strings
            for i in range(len(self.filename)):
                print(f"reading EOS {self.filename[i]}")
                #Charging temperature data
                self.mtpr.append(np.memmap(self.ptm+"eos."+self.filename[i],dtype=np.float32))
                self.mtpr[i] = np.reshape(self.mtpr[i], (2, self.nx,self.ny,self.nz), order="A")
                n_eos = 0
                self.mtpr[i] = self.mtpr[i][n_eos,:,:,:] 
                # n_eos -> 0: temperature ; 1: pressure
                print(f"EOS done {self.filename[i]}")
                print('\n')
                
                #Charging line of sight velocities
                print(f"reading vyy {self.filename[i]}")
                self.mvyy.append(np.memmap(self.ptm+"result_2."+self.filename[i],dtype=np.float32))
                self.mvyy[i] = np.reshape(self.mvyy[i],(self.nx,self.ny,self.nz),order="C")
                print(f"vyy done {self.filename[i]}")
                print('\n')

                #Charging line of sight magnetic field components
                print (f"reading byy {self.filename[i]}")
                self.mbyy.append(np.memmap(self.ptm+"result_6."+self.filename[i],dtype=np.float32))
                self.mbyy[i] = np.reshape(self.mbyy[i],(self.nx,self.ny,self.nz),order="C")
                print(f"byy done {self.filename[i]}")
                print('\n')

                #Charging density values
                print(f"reading rho {self.filename[i]}")
                self.mrho.append(np.memmap(self.ptm+"result_0."+self.filename[i],dtype=np.float32))
                self.mrho[i] = np.reshape(self.mrho[i], (self.nx,self.ny,self.nz), order="A")
                print("scaling...")
                if np.any(self.mrho[i] == 0): #In case this condition happens, the logarithm will diverge, then we will ignore this kind of data by setting all the columns of all quantities related with this value to zero
                    self.mrho[i] = np.reshape(self.mrho[i], (self.nx,self.ny,self.nz), order="A")
                    print(np.shape(self.mrho))
                    print(type(self.mrho))
                    print(f"rho done {self.filename[i]}")
                    print('\n')

                    #x,z coordinates where the density have zero values
                    self.nx0 = np.argwhere(self.mrho[i] == 0)[0][0]
                    self.nz0 = np.argwhere(self.mrho[i] == 0)[0][2]

                    #I set this columns to ones beacuse due to the magnitude of each 
                    #of this measurements, when scaling they are going to be almost negligible
                    #but no zero, therefore there is not going to be any problem when dividing 
                    #mvyy/mrho. It is also done so that this columns are discard from the 
                    #fitting
                    self.mrho[i][self.nx0,:,self.nz0] = np.ones(self.ny)
                    self.mtpr[i][self.nx0,:,self.nz0] = np.ones(self.ny)
                    self.mbyy[i][self.nx0,:,self.nz0] = np.ones(self.ny)
                    self.mvyy[i][self.nx0,:,self.nz0] = np.ones(self.ny)
                    #############################################################
                    #Converting the data into cgs units (if I'm not wrong)
                    #############################################################
                    
                else:
                    self.mvyy[i]=self.mvyy[i]/self.mrho[i]
                    self.mvyy[i] = scaling(self.mvyy[i]) #scaling
                    self.mvyy[i] = np.reshape(self.mvyy[i],(self.nx,self.ny,self.nz),order="C")

                    self.mbyy[i] = scaling(self.mbyy[i])
                    self.mbyy[i] = np.reshape(self.mbyy[i],(self.nx,self.ny,self.nz),order="C")

                #Scaling the charge data    
                print("scaling...")
                self.mvyy[i]=self.mvyy[i]/self.mrho[i]
                self.mvyy[i] = scaling(self.mvyy[i]) #scaling
                self.mvyy[i] = np.reshape(self.mvyy[i],(self.nx,self.ny,self.nz),order="C")

                self.mbyy[i] = scaling(self.mbyy[i])
                self.mbyy[i] = np.reshape(self.mbyy[i],(self.nx,self.ny,self.nz),order="C")

                self.mtpr[i] = scaling(self.mtpr[i])
                self.mtpr[i] = np.reshape(self.mtpr[i], (self.nx,self.ny,self.nz), order="A")
                
                self.mrho[i] = np.log10(self.mrho) #I get the logarithm in base 10 out of the density values so that the big valued data does not damage the code
                self.mrho[i] = scaling(self.mrho[i])
                self.mrho[i] = np.reshape(self.mrho[i], (self.nx,self.ny,self.nz), order="A")

                #Saving ravel outputs for easier splitting
                self.mvyy_ravel.append(ravel_xz(self.mvyy[i]))
                self.mbyy_ravel.append(ravel_xz(self.mbyy[i]))
                self.mtpr_ravel.append(ravel_xz(self.mtpr[i]))
                self.mrho_ravel.append(ravel_xz(self.mrho[i]))



            #Here I am organizing the outputs so that they are
            #organized in one axis over all files and points 
            N_files = len(self.mvyy_ravel)
            N_points = len(self.mvyy_ravel[0])

            def ovf_outputs(input_list): #Organize various files outputs
                N_files = len(input_list)
                N_points = len(input_list[0])
                input_array = np.array(input_list)
                return input_array.reshape(N_files*N_points, self.ny)

            self.mvyy_ravel = ovf_outputs(self.mvyy_ravel)
            self.mbyy_ravel = ovf_outputs(self.mbyy_ravel)
            self.mtpr_ravel = ovf_outputs(self.mtpr_ravel)
            self.mrho_ravel = ovf_outputs(self.mrho_ravel)

        
        self.ravel_inputs = [self.mvyy_ravel, self.mbyy_ravel, self.mtpr_ravel, self.mrho_ravel]
        self.ravel_inputs = np.array(self.ravel_inputs)
        self.ravel_inputs = np.moveaxis(self.ravel_inputs,0,1)
        self.reshaped_inputs = [self.mbyy, self.mvyy, self.mtpr, self.mrho]
        self.reshaped_inputs = np.array(self.reshaped_inputs)
        self.reshaped_inputs = np.moveaxis(self.reshaped_inputs,0,1)
        print(f"*Uploading done*\n")
        return self.ravel_inputs, self.reshaped_inputs
    def charge_intensity(self, ptm, filename):
        self.ptm = ptm
        self.filename = filename
        self.iout = []
        self.iout_ravel = []
        if type(self.filename) == str: #if filename is just a string
            print(f"reading IOUT {self.filename}")
            self.iout = np.memmap(self.ptm+"iout."+self.filename,dtype=np.float32)
            self.iout = np.reshape(self.iout, (self.nx, self.nz), order="A")
            if np.any(self.mrho == 0):
                #x,z coordinates where the density have zero values
                self.nx0 = np.argwhere(self.mrho == 0)[0][0]
                self.nz0 = np.argwhere(self.mrho == 0)[0][2]
                self.iout[self.nx0, self.nz0] = 1 #It is been given this value to the iout pixel
                        #because of the condition given, it is not going to be added in the labels listt
            print("scaling...")
            self.iout = scaling(self.iout) #scaled intensity
            self.iout = np.reshape(self.iout, (self.nx, self.nz), order="A")
            print(np.shape(self.iout))
            self.iout_ravel = self.iout.reshape(self.nx*self.nz)
            print(f"IOUT done {self.filename}")   
            print('\n') 
        else: #if filename is an array of strings
            for i in range(len(self.filename)):
                print(f"reading IOUT {self.filename[i]}")
                self.iout.append(np.memmap(self.ptm+"iout."+self.filename[i],dtype=np.float32))
                self.iout[i] = np.reshape(self.iout[i], (self.nx, self.nz), order="A")
                if np.any(self.mrho[i] == 0):
                    #x,z coordinates where the density have zero values
                    self.nx0 = np.argwhere(self.mrho[i] == 0)[0][0]
                    self.nz0 = np.argwhere(self.mrho[i] == 0)[0][2]

                    self.iout[i][self.nx0, self.nz0] = 1 #It is been given this value to the iout pixel
                            #because of the condition given, it is not going to be added in the labels listt
                print("scaling...")
                self.iout[i] = scaling(self.iout[i]) #scaled intensity
                self.iout[i] = np.reshape(self.iout[i], (self.nx, self.nz), order="A")
                print(np.shape(self.iout[i]))
                print(f"IOUT done {self.filename[i]}")   
                print('\n')
            self.iout_ravel = self.iout.reshape(len(self.iout)*self.nx*self.nz)
            self.iout = np.array(self.iout)
        return self.iout
    def charge_stokes_params(self, stk_ptm, stk_filename, file_type = "nicole"):
        self.stk_ptm = stk_ptm
        self.stk_filename = stk_filename
        self.nlam = 300 #wavelenght interval - its from 6300 amstroengs-
        self.profs = [] #It's for the reshaped data - better for visualization.
        self.profs_ravel = [] #its for the ravel data to make the splitting easier.
        N_profs = 4
        #Charging the stokes profiles for the specific file
        if type(self.stk_filename) == str: #if filename is just a string
            print(f"reading Stokes params {self.stk_filename}")
            for ix in range(self.nx):
                for iy in range(self.nz):
                    p_prof = mpt.read_prof(self.stk_ptm+self.stk_filename, file_type,  self.nx, self.nz, self.nlam, ix, iy)
                    p_prof = np.reshape(p_prof, (self.nlam, N_profs))
                    self.profs.append(p_prof)
            
            self.profs_ravel = np.array(self.profs) #without resizing - better for splitting.
            self.profs = self.profs_ravel.reshape(self.nz, self.nx, self.nlam, N_profs)
        else: #if filename is an array of strings
            profs_interm = []
            for i in range(len(self.stk_filename[i])):
                print(f"reading Stokes params {self.stk_filename[i]}")
                for ix in range(self.nx):
                    for iy in range(self.nz):
                        p_prof = mpt.read_prof(self.stk_ptm+self.stk_filename[i], file_type,  self.nx, self.nz, self.nlam, ix, iy)
                        p_prof = np.reshape(p_prof, (self.nlam, N_profs))
                        profs_interm.append(p_prof)

                profs_interm = np.array(self.profs)
                self.profs_ravel.append(profs_interm) #ravel data

                profs_interm = self.profs.reshape(self.nx, self.nz, self.nlam, N_profs)
                self.profs.append(profs_interm) #resized data

            self.profs_ravel = np.array(self.profs_ravel)
            self.profs = np.array(self.profs)

        return self.profs
    def split_data(self, TRAIN_S, TEST_S):
        """
        Splits the data into a test set and a training set.
        It is a hand made splitting.
        """
        tr_input = []
        te_input = []
        tr_output = []
        te_output = []

	

        








