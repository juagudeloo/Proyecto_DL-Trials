import numpy as np
import matplotlib.pyplot as plt
import model_prof_tools as mprof

def main():
    path = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    file = "052000_0000_0000.prof"
    nx = 480
    ny = 480
    nlam = 300
    prof_im = []
    for i in range(nx):
        for j in range(ny):
            ix, iy = [i,j]
            p_prof = mprof.read_prof(path+file, 'nicole',  nx, ny, nlam, ix, iy)
            p_prof = np.reshape(p_prof, (4, nlam))
            prof_im.append(p_prof)
    
    print(np.shape(prof_im))
    
    im_try = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            im_try[i,j] = prof_im[i+j][0,0]
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(im_try, cmap = 'Greys')
    fig.savefig('Images/I_param_0-052.png')
    
    
if __name__ == "__main__":
    main()
