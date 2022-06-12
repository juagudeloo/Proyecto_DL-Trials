import numpy as np
import matplotlib.pyplot as plt
import model_prof_tools as mprof

def main():
    path = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    file = "052000_0000_0000.prof"
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
            p_prof = np.reshape(p_prof, (N_profs, nlam))
            profs.append(p_prof)
    
    print(np.shape(profs))
    
    #Obtaining the first stokes profile (I) 
    #from the charged data in the 480x480 points
    prof_im_sub = []
    for n in range(N_profs):
        prof_im_sub.append(np.zeros((nx, ny)))
    
    prof_im = []
    for lam in range(nlam):
        prof_im.append(prof_im_sub)
    #Plotting the four profiles for the 300 lambda values
    title = ['I', 'Q', 'U', 'V']
    fig, ax = plt.subplots(300, 4, figsize=(3000,40))
    for lam in range(nlam):
        for n in range(N_profs): 
            for i in range(nx):
                for j in range(ny):
                    prof_im[lam][n][i,j] = profs[i+j][n,lam]
        
        #Plotting the four profiles 
        for n in range(N_profs):
            ax[lam,n].imshow(prof_im[lam][n], cmap = 'Greys')
            ax[lam,n].set_title(title[n])
            
    plt.savefig(f'Images/params_0-052-lam.png')
    
    
if __name__ == "__main__":
    main()
