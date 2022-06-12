import numpy as np
import matplotlib.pyplot as plt
import model_prof_tools as mprof

def main():
    path = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    file = "052000_0000_0000.prof"
    nx = 480
    ny = 480
    nlam = 300
    ix, iy = [0,0]
    prof_im = mprof.read_prof(path+file, 'nicole',  nx, ny, nlam, ix, iy)
    print(np.shape(prof_im))
    fig, ax = plt.subplots(figsize = (10,10))
    x = np.arange(0, len(prof_im), 1)
    ax.scatter(x,prof_im)
    fig.savefig("first_prof.png")
    
    
if __name__ == "__main__":
    main()
