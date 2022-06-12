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
    for i in range(nx):
        for j in range(ny):
            ix, iy = [i,j]
            p_prof = mprof.read_prof(path+file, 'nicole',  nx, ny, nlam, ix, iy)
            p_prof = np.reshape(p_prof, (N_profs, nlam))
            profs.append(p_prof)
    
    print(np.shape(profs))
    
    prof_im = []
    for n in range(N_profs):
        prof_im.append(np.zeros((nx, ny)))
    
    for n in range(N_profs): 
        for i in range(nx):
            for j in range(ny):
                prof_im[n][i,j] = profs[i+j][n,0]
    
    title = ['I', 'Q', 'U', 'V']
    fig, ax = plt.subplots(1, 4, figsize=(10,10))
    for n in range(N_profs):
        ax[n].imshow(prof_im[n], cmap = 'Greys')
        ax[n].set_title(title[n])
    fig.savefig('Images/params_0-052.png')
    
    
if __name__ == "__main__":
    main()
