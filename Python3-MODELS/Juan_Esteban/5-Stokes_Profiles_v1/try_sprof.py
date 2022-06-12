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
    for i in range(0, 3):
        ix, iy = [i,i]
        prof_im.append(mprof.read_prof(path+file, 'nicole',  nx, ny, nlam, ix, iy))
    fig, ax = plt.subplots(2, 2, figsize = (10,10))
    x = np.arange(0, len(prof_im[0]), 1)
    for i in range(len(prof_im)):
        ax[i%2, i].scatter(x,prof_im[i])
    fig.savefig("first_prof.png")
    
    
if __name__ == "__main__":
    main()
