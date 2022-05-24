from cmath import inf, nan
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler

def charge_data(self_ptm:str, self_filename:str, self_nx:int, self_ny:int, self_nz:int):    
    print("*Uploading data...*\n")
    #Scaling function
    def scaling(array):
        scaler = MinMaxScaler()
        array1 = array.reshape(-1,1)
        scaler.fit(array1)
        array1 = scaler.transform(array1)
        array1 = np.ravel(array1)
        return array1
    self_iout = []
    self_mtpr = []
    self_mrho = []
    self_mvyy = []
    self_mbyy = []
    coef = np.sqrt(4.0*np.pi) #for converting data to cgs units
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
    for i in range(len(self_filename)):
        print("reading IOUT")
        self_iout.append(np.memmap(self_ptm+"iout."+self_filename[i],dtype=np.float32))
        self_iout[i] = np.reshape(self_iout[i], (self_nx, self_nz), order="A")
        print("scaling...")
        self_iout[i] = scaling(self_iout[i]) #scaled intensity
        self_iout[i] = np.reshape(self_iout[i], (self_nx, self_nz), order="A")
        print(np.shape(self_iout[i]))
        print("IOUT done")   
        print('\n')

        print("reading EOS")
        self_mtpr.append(np.memmap(self_ptm+"eos."+self_filename[i],dtype=np.float32))
        self_mtpr[i] = np.reshape(self_mtpr[i], (2, self_nx,self_ny,self_nz), order="A")
        n_eos = 0
        self_mtpr[i] = self_mtpr[i][n_eos,:,:,:] 
        print("scaling...")
        self_mtpr[i] = scaling(self_mtpr[i])
        self_mtpr[i] = np.reshape(self_mtpr[i], (self_nx,self_ny,self_nz), order="A")
        # n_eos -> 0: temperature ; 1: pressure
        print("EOS done")
        print('\n')

        #         print("reading vxx")
        #         self.mvxx = np.fromfile(self.ptm+"result_1."+self.filename,dtype=np.float32)
        #         self.mvxx = np.reshape(self.mvxx,(self.nx,self.nz,self.ny),order="C")
        #         print("vxx done")

        print("reading vyy")
        self_mvyy.append(np.memmap(self_ptm+"result_2."+self_filename[i],dtype=np.float32))
        self_mvyy[i] = np.reshape(self_mvyy[i],(self_nx,self_ny,self_nz),order="C")
        print("vyy done")
        print('\n')
        #         print("reading vzz")
        #         self.mvzz = np.fromfile(self.ptm+"result_3."+self.filename,dtype=np.float32)
        #         self.mvzz = np.reshape(self.mvzz,(self.nx,self.nz, self.ny),order="A")
        #         print(np.shape(self.mvzz))
        #         print("vzz done")

        #     print("reading eps")
        #     self.eps = np.fromfile(self.ptm+"result_4."+self.filename,dtype=np.float32)
        #     self.eps = np.reshape(self.eps,(self.nx,self.nz,self.ny),order="C")
        #     print("eps done")

        #         print("reading bxx")
        #         self.mbxx = np.fromfile(self.ptm+"result_5."+self.filename,dtype=np.float32)
        #         self.mbxx = np.reshape(self.mbxx,(self.nx,self.nz,self.ny),order="C")
        #         print("bxx done")

        print ("reading byy")
        self_mbyy.append(np.memmap(self_ptm+"result_6."+self_filename[i],dtype=np.float32))
        self_mbyy[i] = np.reshape(self_mbyy[i],(self_nx,self_ny,self_nz),order="C")
        print("byy done")
        print('\n')

        #         print("reading bzz")
        #         self.mbzz = np.fromfile(self.ptm+"result_7."+self.filename,dtype=np.float32)
        #         self.mbzz = np.reshape(self.mbzz,(self.nx,self.nz, self.ny),order="A")
        #         print(np.shape(self.mbzz))
        #         print("bzz done")

        

        #         self.mvxx=self.mvxx/self.mrho

        print("reading rho")
        self_mrho.append(np.memmap(self_ptm+"result_0."+self_filename[i],dtype=np.float32))
        self_mrho[i] = np.reshape(self_mrho[i], (self_nx,self_ny,self_nz), order="A")
        print("scaling...")
        if np.any(self_mrho[i] == 0):
            self_mrho[i] = np.log10(self_mrho) #I get the logarithm in base 10 out of the density values so that the big valued data does not damage the code
            self_mrho[i] = scaling(self_mrho[i])
            self_mrho[i] = np.reshape(self_mrho[i], (self_nx,self_ny,self_nz), order="A")
            print(np.shape(self_mrho))
            print(type(self_mrho))
            print("rho done")
            print('\n')
        
            nx0 = np.argwhere(self_mrho[i] == 0)[0][0]
            nz0 = np.argwhere(self_mrho[i] == 0)[0][2]
            self_iout[i][nx0, nz0] = nan #It is been given this value to the iout pixel
            #beacuse of the condition given, it is not going to be added in the labels listt
            self_mrho[i][nx0,:,nz0] = np.zeros(self_ny)
            self_mtpr[i][nx0,:,nz0] = np.zeros(self_ny)
            self_mbyy[i][nx0,:,nz0] = np.zeros(self_ny)
            self_mvyy[i][nx0,:,nz0] = np.zeros(self_ny)
        else:
            #############################################################
            #Converting the data into cgs units (if I'm not wrong)
            #############################################################
            self_mvyy[i]=self_mvyy[i]/self_mrho[i]
            self_mvyy[i] = scaling(self_mvyy[i]) #scaling
            self_mvyy[i] = np.reshape(self_mvyy[i],(self_nx,self_ny,self_nz),order="C")
            #         self.mvzz=(self.mvzz/self.mrho)*-6e10
            #         self.mbxx=self.mbxx*coef
            self_mbyy[i]=self_mbyy[i]*coef
            self_mbyy[i] = scaling(self_mbyy[i])
            self_mbyy[i] = np.reshape(self_mbyy[i],(self_nx,self_ny,self_nz),order="C")
            #         self.mbzz=self.mbzz*coef
            
        
    print("*Uploading done*\n")
    return self_iout, self_mbyy, self_mvyy, self_mtpr, self_mrho
    
def plotting(iout, mbyy, mvyy, mrho, mtpr, nx, ny, nz):
    #############################################################
    # Here we select the slices for a nz=cte so that we obtain the 2d graphs of 
    # the intensity, temperature, magnetic field in z and velocity in z
    #############################################################v
    plot_mbyy, plot_iout, plot_mvyy, plot_mrho, plot_mtpr = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz))

    ny_slice = 170 #equivalent in pixels to height
    
    plot_mbyy = mbyy[0][:,ny_slice,:]
    plot_mvyy = mvyy[0][:,ny_slice,:]
    plot_mtpr = mtpr[0][:,ny_slice,:]
    plot_iout = iout[0]

    #############################################################
    # Here we plot the 2D graphs using Matplotlib
    #############################################################
    print("*Plotting...*")
    f = plt.figure(figsize=(15,15))

    ax1 = f.add_subplot(221)
    by = ax1.imshow(plot_mbyy, cmap='gist_gray', origin='lower')
    ax1.set_title('magnetic field in y')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('top', size='5%', pad=0.8)
    f.colorbar(by, cax=cax1, orientation="horizontal", label="NN")
    ax1.grid()

    ax2 = f.add_subplot(222)
    surf = ax2.imshow(plot_iout, cmap='gist_gray', origin='lower')
    ax2.set_title('Intensity')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('top', size='5%', pad=0.8)
    f.colorbar(surf, cax=cax2, orientation="horizontal", label="NN")
    ax2.grid()

    ax3 = f.add_subplot(223)
    divider3 = make_axes_locatable(ax3)
    vy = ax3.imshow(plot_mvyy, cmap='gist_gray', interpolation='gaussian', origin='lower')
    cax3 = divider3.append_axes('top', size='5%', pad=0.8)
    ax3.set_title('Velocity in y')
    f.colorbar(vy, cax=cax3, orientation="horizontal", label="NN")
    ax3.grid()

    ax4 = f.add_subplot(224)

    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('top', size='5%', pad=0.8, label='a')
    data = plot_mtpr
    extent = (np.min(data), np.max(data), np.min(data), np.max(data))
    position = (0,0)      
    tem = ax4.imshow(data, origin='lower',   cmap='gist_gray', interpolation='gaussian', vmax=np.max(data), extent=extent)
    ax4.set_title('Temperature')
    f.colorbar(tem, cax=cax4, orientation="horizontal", label="NN")
    ax4.grid()

    plt.savefig("simulation_images.png")

    print("*Plot done*\n")
#DATA CLASSIFICATION
def data_classif(data, labels, NX, NZ, TR_N, TE_N, PR_N, print_shape = 0):
    #DIVIDING THE DATA WE WANT TO CLASIFFY AND ITS LABELS - It is being used the 
    #scaled data

    print("*Classifying data -> Training, Testing, Predicting*\n")
    print("shape of full data and label arrays:")

    if (print_shape == 1):
        print(f'full data list shape: {np.shape(data)}')
        print(f'full data labels shape: {np.shape(labels)}\n')

    #TRAINING SET AND DATA SET
    tr_data = []
    tr_labels = []

    print("training data classification...")
    #Here I'm dividing the whole data for the nx, nz dimensions. The first half is the training set and the second half is the test set
    for i in range(TR_N):
        for j in range(NX):
            for k in range(NZ):
                if not np.all(data[i][0][j,:,k] == 0):
                    tr_data.append([data[i][0][j,:,k], 
                                    data[i][1][j,:,k],
                                    data[i][2][j,:,k],
                                    data[i][3][j,:,k]]) #It puts the magnetic field, velocity, temperature and density values in one row for 240x240=57600 columns
                    tr_labels.append(labels[i][j,k]) #the values from the column of targets
                    
    tr_data = np.array(tr_data)   
    tr_labels = np.array(tr_labels)  
    print("Done")

    print("shape of training data and training labels:")
    if (print_shape == 1):
        print(f'training data shape: {np.shape(tr_data)}')
        print(f'training labels shape: {np.shape(tr_labels)}\n')

    print("test data classification...")
    te_data = []
    te_labels = []
    for i in range(TE_N):
        for j in range(NX):
            for k in range(NZ):
                if not np.all(data[TR_N + i][0][j,:,k]==0):
                    te_data.append([data[TR_N + i][0][j,:,k], 
                                    data[TR_N + i][1][j,:,k],
                                    data[TR_N + i][2][j,:,k],
                                    data[TR_N + i][3][j,:,k]]) #It puts the magnetic field, velocity, temperature and density values in one row for 240x240=57600 columns
                    te_labels.append(labels[TR_N + i][j,k]) #the values from the column of targets

    te_data = np.array(te_data)   
    te_labels = np.array(te_labels)  
    print("Done")
    
    print("shape of testing data and testing labels:")
    if (print_shape == 1):
        print(f'test data shape: {np.shape(te_data)}')
        print(f'test labels shape: {np.shape(te_labels)}\n')


    print("predicting data classification...")
    pr_data = []
    pr_labels = []
    PR_INIC = TR_N + TE_N
    print(f"PR_INIC + PR_N = {PR_INIC + PR_N}")
    if PR_N>1:
        for i in range(PR_N):
            for j in range(NX):
                for k in range(NZ):
                    if not np.all(data[PR_INIC + i][0][j,:,k]==0):
                        pr_data.append([data[PR_INIC + i][0][j,:,k], 
                                        data[PR_INIC + i][1][j,:,k],
                                        data[PR_INIC + i][2][j,:,k],
                                        data[PR_INIC + i][3][j,:,k]]) #It puts the magnetic field, velocity, temperature and density values in one row for 240x240=57600 columns
                        pr_labels.append(labels[PR_INIC + i][j,k]) #the values from the column of targets
                    else:
                        raise ValueError('sorry, you have encountered prediction values that are not predictable')
    else:
        for j in range(NX):
                for k in range(NZ):
                    if not np.all(data[(PR_INIC + PR_N) - 1][0][j,:,k]==0):
                        pr_data.append([data[(PR_INIC + PR_N) - 1][0][j,:,k], 
                                        data[(PR_INIC + PR_N) - 1][1][j,:,k],
                                        data[(PR_INIC + PR_N) - 1][2][j,:,k],
                                        data[(PR_INIC + PR_N) - 1][3][j,:,k]]) #It puts the magnetic field, velocity, temperature and density values in one row for 240x240=57600 columns
                        pr_labels.append(labels[(PR_INIC + PR_N) - 1][j,k])
                    else:
                        raise ValueError('sorry, you have encountered prediction values that are not predictable')

    pr_data = np.array(pr_data)    
    pr_labels = np.array(pr_labels)  
    print("Done")
    
    print("shape of predicting data and predicting labels:")
    if (print_shape == 1):
        print(f'predict data shape: {np.shape(pr_data)}')
        print(f'predict labels shape: {np.shape(pr_labels)}\n')

    print("*Clasifying data done*")

    return tr_data, tr_labels, te_data, te_labels, pr_data, pr_labels

#AFTER-TRAINING FUNCTIONS
def plot_dist(intensity_out, history, var_metrics, titles, PR_L, dist_fig_name, error_fig_name, var_values=0, plot_var_values=True, plot_errors = True, loss_metric = 'mean_squared_error'):
    """
    ----------------------------------------------------------------------------
    Distribution and error relation plots
    ----------------------------------------------------------------------------
    intensity_out: array of the intensity predicted by the models
    history: array with the time information the training
    plot_var_values: 
    var_values: like the different values of the learning rate
    var_metrics: list with the time metrics of the testing step
    titles: list of strings with the title names of the distribution plots
    PR_L: Comparison data for the predicting values 
    dist_fig_name: Distribution plot file name
    error_fig_name: Error relation file name
    """
    path = "Images/"
    print("*plotting...*")
    diff = 0
    row_len = len(intensity_out)
    if plot_errors == True:
        if plot_var_values is True:
            fig1, ax1 = plt.subplots(3, row_len, figsize = (40,10))
            for i in range(row_len):
                diff = np.absolute(np.ravel(intensity_out[i])-np.ravel(PR_L))/np.absolute(np.ravel(PR_L))
                ax1[0,i].hist(x=diff, bins = 'auto',
                                            alpha=0.7, 
                                            rwidth=0.85)
                ax1[0,i].set_title(titles[i])
            for i in range(row_len):
                ax1[1,i].imshow(intensity_out[i], cmap = 'gist_gray')
                ax1[1,i].set_title(titles[i])

            x = np.arange(1, 8)+1
            for i in range(row_len):
                y = history[i].history[loss_metric][1:10]
                print(np.size(y))
                print(np.size(x))
                ax1[2,i].scatter(x, y)
                ax1[2,i].set_ylabel(loss_metric)
                ax1[2,i].set_xlabel('epoch')
                ax1[2,i].set_ylim(0,0.3)
            fig1.savefig(path+dist_fig_name)
            print("*figure saved*")

            #Error distribution
            final_error = np.zeros(row_len)
            for i in range(row_len):
                y = history[i].history[loss_metric]
                final_error[i] = history[i].history[loss_metric][len(y) - 1]
            x = np.arange(0,row_len)
            fig3, ax3 = plt.subplots(figsize = (5,5))

            ax3.plot(x, np.log10(final_error), label ='final_training_error')
            ax3.plot(x, np.log10(var_values), label = 'learning_rate')
            ax3.plot(x, np.log10(var_metrics), label = 'test metric')
            ax3.legend()
            fig3.savefig(path+error_fig_name)
        else:
            fig1, ax1 = plt.subplots(3, row_len, figsize = (40,10))
            for i in range(row_len):
                diff = np.absolute(np.ravel(intensity_out[i])-np.ravel(PR_L))/np.absolute(np.ravel(PR_L))
                ax1[0,i].hist(x=diff, bins = 'auto',
                                            alpha=0.7, 
                                            rwidth=0.85)
                ax1[0,i].set_title(titles[i])
            for i in range(row_len):
                ax1[1,i].imshow(intensity_out[i], cmap = 'gist_gray')
                ax1[1,i].set_title(titles[i])

            x = np.arange(1, 8)+1
            for i in range(row_len):
                y = history[i].history[loss_metric][1:10]
                print(np.size(y))
                print(np.size(x))
                ax1[2,i].scatter(x, y)
                ax1[2,i].set_ylabel(loss_metric)
                ax1[2,i].set_xlabel('epoch')
                ax1[2,i].set_ylim(0,0.3)
            fig1.savefig(path+dist_fig_name)
            print("*figure saved*")

            #Error distribution
            final_error = np.zeros(row_len)
            for i in range(row_len):
                y = history[i].history[loss_metric]
                final_error[i] = history[i].history[loss_metric][len(y) - 1]
            x = np.arange(0,row_len)
            fig3, ax3 = plt.subplots(figsize = (5,5))

            ax3.plot(x, np.log10(final_error), label ='final_training_error')
            ax3.plot(x, np.log10(var_metrics), label = 'test metric')
            ax3.legend()
            fig3.savefig(path+error_fig_name)
    else:
        fig1, ax1 = plt.subplots(3, row_len, figsize = (40,10))
        for i in range(row_len):
            diff = np.absolute(np.ravel(intensity_out[i])-np.ravel(PR_L))/np.absolute(np.ravel(PR_L))
            ax1[0,i].hist(x=diff, bins = 'auto',
                                        alpha=0.7, 
                                        rwidth=0.85)
            ax1[0,i].set_title(titles[i])
        for i in range(row_len):
            ax1[1,i].imshow(intensity_out[i], cmap = 'gist_gray')
            ax1[1,i].set_title(titles[i])

        x = np.arange(1, 8)+1
        for i in range(row_len):
            y = history[i].history[loss_metric][1:10]
            print(np.size(y))
            print(np.size(x))
            ax1[2,i].scatter(x, y)
            ax1[2,i].set_ylabel(loss_metric)
            ax1[2,i].set_xlabel('epoch')
            ax1[2,i].set_ylim(0,0.3)
        fig1.savefig(path+dist_fig_name)
        print("*figure saved*")
        

