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
from torch.utils.data import TensorDataset

import os

from tqdm import tqdm 

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#####################################################################################################################
# Data class
#####################################################################################################################

class MuRAM():
    #To rescale all the data e are going to use a max values of the order of the general maximum value of the data, and 
    def __init__(self, ptm:str, filenames:list):
        self.ptm = ptm
        self.filenames = filenames
        self.nlam = 300 #this parameter is useful when managing the Stokes parameters #wavelenght interval - its from 6300 amstroengs in steps of 10 amstroengs
        self.nx = 480
        self.ny = 256 #height axis
        self.nz = 480
    def charge_quantities(self, filename, scale = True, opt_depth_stratif = True, opt_len = 20, vertical_comp = True):
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
        Reading {filename} MuRAM data...
        ######################## 
              """)

        print("Charging temperature ...")
        mtpr = np.load(self.ptm+"mtpr_"+filename+".npy").flatten()

        print("Charging magnetic field vector...")
        mbxx = np.load(self.ptm+"mbxx_"+filename+".npy")
        mbyy = np.load(self.ptm+"mbyy_"+filename+".npy")
        mbzz = np.load(self.ptm+"mbzz_"+filename+".npy")

        coef = np.sqrt(4.0*np.pi) #cgs units conversion300

        mbxx=mbxx*coef
        mbyy=mbyy*coef
        mbzz=mbzz*coef
        
        mbqq = np.sign(mbxx**2 - mbzz**2)*np.sqrt(np.abs(mbxx**2 - mbzz**2))
        mbuu = np.sign(mbxx*mbzz)*np.sqrt(np.abs(mbxx*mbzz))
        mbvv = mbyy

        print("Charging density...")
        mrho = np.load(self.ptm+"mrho_"+filename+".npy")

        print("Charge velocity...")
        mvxx = np.load(self.ptm+"mvxx_"+filename+".npy")
        mvyy = np.load(self.ptm+"mvyy_"+filename+".npy")
        mvzz = np.load(self.ptm+"mvzz_"+filename+".npy")

        mvxx = mvxx/mrho
        mvyy = mvyy/mrho
        mvzz = mvzz/mrho
        
        """
        Narray of the atmosphere quantities...
        """
        
        if vertical_comp:
            mags_names = ["T", "rho", "Bv", "vy"]
            atm_quant = np.array([mtpr, mrho, mbvv, mvyy])
        else:
            
            mags_names = ["T", "rho", "Bq", "Bu", "Bv", "vy"]
            atm_quant = np.array([mtpr, mrho, mbqq, mbuu, mbvv, mvyy])
        atm_quant = np.moveaxis(atm_quant, 0,1)
        
        atm_quant = np.memmap.reshape(atm_quant, (self.nx, self.ny, self.nz, atm_quant.shape[1]))

        print("Charging Stokes vectors...")
        stokes = np.load(self.ptm+filename+"_prof.npy")
        
        if opt_depth_stratif:
            print("Applying optical depth stratification...")
            opt_depth = np.load(self.ptm+"optical_depth_"+filename+".npy")
            #optical depth points
            tau_out = self.ptm+"array_of_tau_"+filename+f"_{opt_len}_depth_points.npy"
            tau = np.linspace(-3, 1, opt_len)
            np.save(tau_out, tau)

            #optical stratification
            opt_mags_interp = {}
            opt_mags = np.zeros((self.nx, opt_len, self.nz, atm_quant.shape[-1]))
            opt_mags_out =self.ptm+"optical_stratified_atm_modified_mbvuq_"+filename+f"_{opt_len}_depth_points_{atm_quant.shape[-1]}_components.npy"
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
            
            self.phys_maxmin = {}
            self.phys_maxmin["T"] = [2e4, 0]
            self.phys_maxmin["B"] = [3e3, -3e3]
            self.phys_maxmin["Rho"] = [1e-5, 1e-10]
            self.phys_maxmin["V"] = [1e6, -1e6]

            mtpr = norm_func(mtpr, self.phys_maxmin["T"])

            mbqq = norm_func(mbqq, self.phys_maxmin["B"])
            mbuu = norm_func(mbuu, self.phys_maxmin["B"])
            mbvv = norm_func(mbvv, self.phys_maxmin["B"])

            mrho = norm_func(mrho, self.phys_maxmin["Rho"])

            mvxx = norm_func(mvxx, self.phys_maxmin["V"])
            mvyy = norm_func(mvyy, self.phys_maxmin["V"])
            mvzz = norm_func(mvzz, self.phys_maxmin["V"] )
            
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
        
        #Resampling to less spectral points using a gaussian kernel


        new_points = 36
        new_stokes_out = self.ptm+"resampled_stokes_"+f"f{filename}_{new_points}_wl_points.npy"
        if not os.path.exists(new_stokes_out):
            N_kernel_points = 13 # number of points of the kernel.
            def gauss(n=N_kernel_points,sigma=1):
                r = range(-int(n/2),int(n/2)+1)
                return np.array([1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r])
            g = gauss()

            new_wl = np.linspace(0,288,36, dtype=np.int64)
            new_wl = np.add(new_wl, 6)
            new_stokes = np.zeros((self.nx, self.nz, new_points, stokes.shape[-1]))

            for s in range(len(self.stokes_maxmin)):
                for ix in tqdm(range(self.nx)):
                    for iz in range(self.nz):
                        spectrum = stokes[ix,iz,:,s]
                        resampled_spectrum = np.zeros(new_points)
                        i = 0
                        for center_wl in new_wl:
                            low_limit = center_wl-6
                            upper_limit = center_wl+7

                            if center_wl == 6:
                                shorten_spect = spectrum[0:13]
                            elif center_wl == 294:
                                shorten_spect = spectrum[-14:-1]
                            else:
                                shorten_spect = spectrum[low_limit:upper_limit]

                            resampled_spectrum[i] = np.sum(np.multiply(shorten_spect,g))
                            i += 1
                        new_stokes[ix,iz,:,s] = resampled_spectrum
            np.save(new_stokes_out, new_stokes)
        else:
            new_stokes = np.load(new_stokes_out)


        verbose = 0
        #if verbose:
        #    print(f""" STOKES:
        #    I_max = {np.max(stokes[:,:,:,0])}
        #    Q_max = {np.max(stokes[:,:,:,1])}
        #    U_max = {np.max(stokes[:,:,:,2])}
        #    V_max = {np.max(stokes[:,:,:,3])}
        #    I_min = {np.min(stokes[:,:,:,0])}
        #    Q_min = {np.min(stokes[:,:,:,1])}
        #    U_min = {np.min(stokes[:,:,:,2])}
        #    V_min = {np.min(stokes[:,:,:,3])}
        #    """)
#
        #    print(f"""
        #    MAX VALUES:
        #    mtpr max = {np.max(mtpr)}
        #    mbxx max = {np.max(mbxx)}
        #    mbyy max = {np.max(mbyy)}
        #    mbzz max = {np.max(mbzz)}
        #    mrho max = {np.max(mrho)}
        #    mvxx max = {np.max(mvxx)}
        #    mvyy max = {np.max(mvyy)}
        #    mvzz max = {np.max(mvzz)}
        #        """)
#
        #    print(f"""
        #    MIN VALUES:
        #    mtpr min = {np.min(mtpr)}
        #    mbxx min = {np.min(mbxx)}
        #    mbyy min = {np.min(mbyy)}
        #    mbzz min = {np.min(mbzz)}
        #    mrho min = {np.min(mrho)}
        #    mvxx min = {np.min(mvxx)}
        #    mvyy min = {np.min(mvyy)}
        #    mvzz min = {np.min(mvzz)}
        #        """)

        print(f"""
        ######################## 
        {filename} MuRAM data charged...
        ######################## 
              """)
        return atm_quant, new_stokes
    def granular_intergranular(self, filename, gran_inter_zones = False, scale = True, opt_depth_stratif = True, opt_len = 20, vertical_comp = True):
        """
        Function for leveraging the granular and intergranular zones pixels.
        ====================================================================
        gran_inter_zones: Boolean. If True, it will give the data separated on granular and intergranular. 
        Default is False for training.
        """
        #Charging the physicial magnitudes and the stokes parameters
        atm_quant, stokes = self.charge_quantities(filename, scale = scale, opt_depth_stratif = opt_depth_stratif, opt_len = opt_len, vertical_comp = vertical_comp)
        
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
        
    def train_test_sets(self, gran_inter_zones = False, scale = True, opt_depth_stratif = True, opt_len = 20, vertical_comp = True):
        all_atm_quant = []
        all_stokes = []

        for fln in self.filenames:
            atm_quant, stokes = self.granular_intergranular(fln, gran_inter_zones, scale, opt_depth_stratif, opt_len, vertical_comp)
            all_atm_quant.append(atm_quant)
            all_stokes.append(stokes)

        all_atm_quant = np.concatenate(all_atm_quant, axis=0)
        all_stokes = np.concatenate(all_stokes, axis=0)
        print(all_atm_quant.shape, all_stokes.shape)
        print("splitting...")
        in_train, in_test, out_train, out_test = train_test_split(all_stokes, all_atm_quant, test_size=0.33, random_state=42)

        # Setup device agnostic code
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Tensors stored in: {device}")
        
        fig, ax = plt.subplots(2,6, figsize=(5*6,5))
        for i in range(6):
            ax[0,i].plot(range(36), in_train[0,:,i])
            ax[1,i].plot(range(36), in_test[0,:,i])
        plt.show()
        
        fig, ax = plt.subplots(2,6, figsize=(5*6,5))
        for i in range(4):
            ax[0,i].plot(range(20), out_train[0,:,i])
            ax[1,i].plot(range(20), out_test[0,:,i])
        plt.show()
        
        in_train = torch.from_numpy(in_train).to(device)
        in_test = torch.from_numpy(in_test).to(device)
        out_train = torch.from_numpy(out_train).to(device)
        out_test = torch.from_numpy(out_test).to(device)

        out_train = torch.reshape(out_train, (out_train.size()[0], out_train.size()[1]*out_train.size()[2]))
        out_test = torch.reshape(out_test, (out_test.size()[0], out_test.size()[1]*out_test.size()[2]))
        print(f"""
        Shape of the data
                tr_input shape ={in_train.size()}
                test_input shape = {in_test.size()}
                tr_output shape = {out_train.size()}
                test_output shape = {out_test.size()}
                    """)
        #Train and test dataloader
        train_dataset = TensorDataset(in_train.to(device), out_train.to(device))
        test_dataset = TensorDataset(in_test.to(device), out_test.to(device))

        return train_dataset, test_dataset
