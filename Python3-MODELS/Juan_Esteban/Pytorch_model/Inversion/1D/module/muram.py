# %% [code]
# %% [code]
# %% [code]
#####################################################################################################################
# Libraries
#####################################################################################################################
import numpy as np
from skimage import filters
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import torch
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#####################################################################################################################
# Data class
#####################################################################################################################

class MuRAM():
    #To rescale all the data e are going to use a max values of the order of the general maximum value of the data, and 
    def __init__(self, ptm:str, filename:str):
        """
        ptm (str): Path for MURAM data.
        """
        self.ptm = ptm
        self.filename = filename
        self.nlam = 300 #this parameter is useful when managing the Stokes parameters #wavelenght interval - its from 6300 amstroengs in steps of 10 amstroengs
        self.nx = 480
        self.ny = 256
        self.nz = 480
    def charge_quantities(self, scale = True, opt_depth_stratif = True, opt_len = 20, vertical_comp = True):
        """
        Function for charging both atmosphere physical quantities and the Stokes parameters
        scale: Boolean. Decides wether or not to scale the data. Default is True.
        opt_depth_stratif: Boolean. Decides whether or not to stratify the atmosphere as optical depth or keep the geometrical height. Default is True.
        opt_len: Number of points of the optical depth stratification (just works if opt_depth_stratif == True).
        """
        #To rescale all the data e are going to use a max values of the order of the general maximum value of the data, and 
        def norm_func(arr, maxmin):
            max_val = maxmin[0]
            min_val = maxmin[1]
            return (arr-min_val)/(max_val-min_val)

        print(f"""
        ######################## 
        Reading {self.filename} MuRAM data...
        ######################## 
              """)

        print("Charging temperature ...")
        mtpr = np.load(self.ptm+"mtpr_"+self.filename+".npy").flatten()

        print("Charging magnetic field vector...")
        mbxx = np.load(self.ptm+"mbxx_"+self.filename+".npy")
        mbyy = np.load(self.ptm+"mbyy_"+self.filename+".npy")
        mbzz = np.load(self.ptm+"mbzz_"+self.filename+".npy")

        coef = np.sqrt(4.0*np.pi) #cgs units conversion300

        mbxx=mbxx*coef
        mbyy=mbyy*coef
        mbzz=mbzz*coef

        print("Charging density...")
        mrho = np.load(self.ptm+"mrho_"+self.filename+".npy")

        print("Charge velocity...")
        mvxx = np.load(self.ptm+"mvxx_"+self.filename+".npy")
        mvyy = np.load(self.ptm+"mvyy_"+self.filename+".npy")
        mvzz = np.load(self.ptm+"mvzz_"+self.filename+".npy")

        mvxx = mvxx/mrho
        mvyy = mvyy/mrho
        mvzz = mvzz/mrho
        
        if scale:
            print("Scaling...")

            self.phys_maxmin = {}
            self.phys_maxmin["T"] = [1e4, 0]
            self.phys_maxmin["B"] = [1e3, -1e3]
            self.phys_maxmin["Rho"] = [1e-5, 1e-12]
            self.phys_maxmin["V"] = [1e6, -1e6]

            mtpr = norm_func(mtpr, self.phys_maxmin["T"])

            mbxx = norm_func(mbxx, self.phys_maxmin["B"])
            mbyy = norm_func(mbyy, self.phys_maxmin["B"])
            mbzz = norm_func(mbzz, self.phys_maxmin["B"])

            mrho = norm_func(mrho, self.phys_maxmin["Rho"])

            mvxx = norm_func(mvxx, self.phys_maxmin["V"])
            mvyy = norm_func(mvyy, self.phys_maxmin["V"])
            mvzz = norm_func(mvzz, self.phys_maxmin["V"] )
        
        """
        Narray of the atmosphere quantities...
        """
        if vertical_comp:
            mags_names = ["T", "rho", "By", "vy"]
            atm_quant = np.array([mtpr, mrho, mbyy, mvyy])
        else:
            mags_names = ["T", "rho", "Bx", "By", "Bz", "vx", "vy", "vz"]
            atm_quant = np.array([mtpr, mrho, mbxx, mbyy, mbzz, mvxx, mvyy, mvzz])
        atm_quant = np.moveaxis(atm_quant, 0,1)
        atm_quant = np.memmap.reshape(atm_quant, (self.nx, self.ny, self.nz, atm_quant.shape[1]))

        print("Charging Stokes vectors...")
        stokes = np.load(self.ptm+self.filename+"_prof.npy")
        
        if opt_depth_stratif:
            print("Applying optical depth stratification...")
            opt_depth = np.load(self.ptm+"optical_depth_"+self.filename+".npy")
            #optical depth points
            optical_dir = self.ptm+"OpticalStratification/"
            if not os.path.exists(optical_dir):
                os.mkdir(optical_dir)
                
            tau_out = self.ptm+"OpticalStratification/"+"array_of_tau_"+self.filename+f"_{opt_len}_depth_points.npy"
            if not os.path.exists(tau_out):
                tau = np.linspace(-3, 1, opt_len)
                np.save(tau_out, tau)

            #optical stratification
            opt_mags_interp = {}
            opt_mags = np.zeros((self.nx, opt_len, self.nz, atm_quant.shape[-1]))
            opt_mags_out =optical_dir+"optical_stratified_atm_"+self.filename+f"_{opt_len}_depth_points.npy"
            if not os.path.exists(opt_mags_out):
                for ix in tqdm(range(self.nx)):
                        for iz in range(self.nz):
                            for i in range(len(mags_names)):
                                opt_mags_interp[mags_names[i]] = interp1d(opt_depth[ix,:,iz], atm_quant[ix,:,iz,i])
                                opt_mags[ix,:,iz,i] = opt_mags_interp[mags_names[i]](tau)
                np.save(opt_mags_out, opt_mags)
            else:
                opt_mags = np.load(opt_mags_out)
            atm_quant = opt_mags
            print(opt_mags.shape)

        if scale:
            print("Scaling...")
            self.stokes_maxmin = {}
            #Stokes I
            self.stokes_maxmin["I"] = [1e14, 0]
            stokes[:,:,:,0] = norm_func(stokes[:,:,:,0], self.stokes_maxmin["I"])

            self.stokes_maxmin["Q"] = [1e14, -1e14]
            stokes[:,:,:,1] = norm_func(stokes[:,:,:,1], self.stokes_maxmin["Q"])


            self.stokes_maxmin["U"] = [1e14, -1e14]
            stokes[:,:,:,2] = norm_func(stokes[:,:,:,2], self.stokes_maxmin["U"])


            self.stokes_maxmin["V"] = [1e14, -1e14]
            stokes[:,:,:,3] = norm_func(stokes[:,:,:,3], self.stokes_maxmin["I"])



        print(f""" STOKES:
        I_max = {np.max(stokes[:,:,:,0])}
        Q_max = {np.max(stokes[:,:,:,1])}
        U_max = {np.max(stokes[:,:,:,2])}
        V_max = {np.max(stokes[:,:,:,3])}
        I_min = {np.min(stokes[:,:,:,0])}
        Q_min = {np.min(stokes[:,:,:,1])}
        U_min = {np.min(stokes[:,:,:,2])}
        V_min = {np.min(stokes[:,:,:,3])}
        """)

        print(f"""
        MAX VALUES:
        mtpr max = {np.max(mtpr)}
        mbxx max = {np.max(mbxx)}
        mbyy max = {np.max(mbyy)}
        mbzz max = {np.max(mbzz)}
        mrho max = {np.max(mrho)}
        mvxx max = {np.max(mvxx)}
        mvyy max = {np.max(mvyy)}
        mvzz max = {np.max(mvzz)}
              """)

        print(f"""
        MIN VALUES:
        mtpr min = {np.min(mtpr)}
        mbxx min = {np.min(mbxx)}
        mbyy min = {np.min(mbyy)}
        mbzz min = {np.min(mbzz)}
        mrho min = {np.min(mrho)}
        mvxx min = {np.min(mvxx)}
        mvyy min = {np.min(mvyy)}
        mvzz min = {np.min(mvzz)}
              """)


        print(f"""
        ######################## 
        {self.filename} MuRAM data charged...
        ######################## 
              """)

        return atm_quant, stokes
    def granular_intergranular(self, gran_inter_zones = False, scale = True, opt_depth_stratif = True, opt_len = 20, vertical_comp = True):
        """
        Function for leveraging the granular and intergranular zones pixels.
        ====================================================================
        gran_inter_zones: Boolean. If True, it will give the data separated on granular and intergranular. 
        Default is False for training.
        """
        #Charging the physicial magnitudes and the stokes parameters
        atm_quant, stokes = self.charge_quantities(scale = scale, opt_depth_stratif = opt_depth_stratif, opt_len = opt_len, vertical_comp = vertical_comp)
        # Originally the MURAM code takes y as the vertical axis. However, here we are changing it and leaving it as x,y,z with z the vertical axis.
        atm_quant = np.moveaxis(atm_quant,1,2) #480,480,256
        
        print("Separating in granular and intergranular...")
        #Selecting the continuum for identifying the granular and intergranular zones.
        I_cont = stokes[:,:,0,0]
        
        #Applying an otsu filter for the separation of the zones.
        thresh1 = filters.threshold_otsu(I_cont)
        im_bin = I_cont<thresh1
        gran_mask =  np.ma.masked_array(I_cont, mask=im_bin).mask
        inter_mask = np.ma.masked_array(I_cont, mask=~im_bin).mask
        
        #Applying the obtained mask on the data.
        atm_quant_gran = atm_quant[gran_mask]
        atm_quant_inter = atm_quant[inter_mask]
        stokes_gran = stokes[gran_mask]
        stokes_inter = stokes[inter_mask]
        len_inter = atm_quant_inter.shape[0]
        len_gran = atm_quant_gran.shape[0]
        
        #leveraging the quantity of data from the granular and intergranular zones by a random dropping of elements of the greater zone.
        if not gran_inter_zones:
            print("leveraging...")
            index_select  = []
            np.random.seed(50)
            if len_inter < len_gran:
                index_select = np.random.choice(range(len_gran), size = (len_inter,), replace = False)
                atm_quant_leveraged = np.concatenate((atm_quant_gran[index_select], atm_quant_inter), axis = 0)
                stokes_leveraged = np.concatenate((stokes_gran[index_select], stokes_inter), axis = 0)
            elif len_inter > len_gran:
                index_select = np.random.choice(range(len_inter), size = (len_gran,), replace = False)
                atm_quant_leveraged = np.concatenate((atm_quant_gran, atm_quant_inter[index_select]), axis = 0)
                stokes_leveraged = np.concatenate((stokes_gran, stokes_inter[index_select]), axis = 0)
            print("Done")
            return atm_quant_leveraged, stokes_leveraged
        else:
            print("Done! Returning the granular and intergraular zones quantities.")
            return atm_quant_gran, atm_quant_inter, stokes_gran, stokes_inter
        
    def train_test_sets(self, name_of_input, gran_inter_zones = False, scale = True, opt_depth_stratif = True, opt_len = 20, vertical_comp = True):
        atm_quant, stokes = self.granular_intergranular(gran_inter_zones = gran_inter_zones, scale = scale, opt_depth_stratif = opt_depth_stratif, opt_len = opt_len, vertical_comp = vertical_comp)
        print("splitting...")
        random_seed = 42
        if name_of_input == "Stokes":
            in_train, in_test, out_train, out_test = train_test_split(stokes, atm_quant, test_size=0.33, random_state=random_seed)
        elif name_of_input == "Atm":
            in_train, in_test, out_train, out_test = train_test_split(stokes, atm_quant, test_size=0.33, random_state=random_seed)
        else:
            raise ValueError("Not possible input")
            
        # Setup device agnostic code
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Tensors stored in: {device}")

        in_train = torch.from_numpy(in_train).to(device)
        in_test = torch.from_numpy(in_test).to(device)
        out_train = torch.from_numpy(out_train).to(device)
        out_test = torch.from_numpy(out_test).to(device)

        return in_train, in_test, out_train, out_test
