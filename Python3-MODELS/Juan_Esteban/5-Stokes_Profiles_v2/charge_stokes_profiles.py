import numpy as np
import matplotlib.pyplot as plt
import model_prof_tools as mprof

def main():
    path = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    file = "053000_0000_0000.prof"
    nx = 480
    ny = 480
    nlam = 300
    profs = []
    N_profs = 4
    #Charging the stokes profiles for the specific file
    for i in range(nx):
        for j in range(ny):
            ix, iy = [i,j]
            p_prof = mprof.read_prof(path+file, 'nicole',  nx, ny, nlam, ix, iy)
            p_prof = np.reshape(p_prof, (nlam, N_profs))
            profs.append(p_prof)
    
    print(np.shape(profs))
    
    #Obtaining the stokes profiles
    #from the charged data in the 480x480 points
    #for a specific wavelength step value
    prof_im = []
    for n in range(N_profs):
        prof_im.append(np.zeros((nx, ny)))
    print(np.shape(prof_im))
    print(np.shape(prof_im[0][0,0]))
    print(np.shape(prof_im[:][0][0]))
    print(np.shape(prof_im[0][:][0]))
    print(np.shape(prof_im[0][0][:]))     

    lam = 60 #wavelength step number
    for n in range(N_profs): 
        for i in range(nx):
            for j in range(ny):
                prof_im[n][i,j] = profs[i*ny+j][lam,n]
    
    #Plotting the four profiles 
    title = ['I', 'Q', 'U', 'V']
    fig, ax = plt.subplots(1, 4, figsize=(40,10))
    for n in range(N_profs):
        ax[n].imshow(prof_im[n], cmap = 'gist_gray')
        ax[n].set_title(title[n])
    fig.savefig(f'Images/params_0-053-lam_{lam}.png')
    
    ix = 200
    iy =14
    fig2, ax2 = plt.subplots(1,4,figsize = (40,10))
    for n in range(N_profs):
        ax2[n].plot(range(nlam), profs[ix*ny+iy][:,n])
    fig2.savefig('profiles_plots.png')    
    
if __name__ == "__main__":
    main()
