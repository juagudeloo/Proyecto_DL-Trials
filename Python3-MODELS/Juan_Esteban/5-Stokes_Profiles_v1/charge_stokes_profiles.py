import numpy as np
import matplotlib.pyplot as plt 
import model_prof_tools as mprof

def main():
    #extracting parameters
    path = "/mnt/scratch/juagudeloo/Stokes_profiles/PROFILES/"
    filename = path+"052000_0000_0000.prof"
    filetype = "nicole"
    nlam = 300
    sequential = 1
    nx, ny = [480, 480]
    ix, iy = [0,0]

    profiles = mprof.read_prof(filename, filetype, nx, ny, nlam, ix, iy, sequential)
    print(np.shape(profiles))
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(range(len(profiles)), profiles)
    fig.savefig("profiles.png")

    
if __name__ == "__main__":
    main()