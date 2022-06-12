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
    #Charging the stokes profiles 
    for i in range(nx):
        for j in range(ny):
            ix, iy = [i,j]
            p_prof = mprof.read_prof(path+file, 'nicole',  nx, ny, nlam, ix, iy)
            p_prof = np.reshape(p_prof, (N_profs, nlam))
            profs.append(p_prof)
    
    print(np.shape(profs))
    
    ix = 200
    iy = 200
    nlam_inf = 6300.5
    nlam_sup = nlam_inf+nlam*0.01
    x = np.linspace(nlam_inf,nlam_sup,nlam)
    
    title = ['I', 'Q', 'U', 'V']
    fig, ax = plt.subplots(1,4, figsize = (40, 10))
    for prof in range(N_profs):
        ax[prof].plot(x, p_prof[ix+iy][prof, :])
        ax.set_title(title[prof])
    fig.savefig(f'Images/profiles-ix_{ix}-iy_{iy}.png')
    
if __name__ == "__main__":
    main()
