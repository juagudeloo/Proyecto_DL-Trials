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
    profiles = np.array(profiles)
    print(np.shape(profiles))
    
    
    fig1, ax1 = plt.subplots(figsize = (10,10))
    ax1.scatter(range(len(profiles)), profiles[:])
    ax1.set_ylim((0,1))
    ax1.set_xlim((0,1200))
    fig1.savefig("wholw_array.png")
    
    profiles = np.reshape(4, 300)
    
    title = ['I', 'U', 'V', 'Q']
    
    fig2, ax2 = plt.subplots(1,4,figsize=(40,10))
    N_profiles = len(profiles[:,300])
    for i in range(N_profiles):
        ax2[i].scatter(range(profiles[i,:]), profiles[i,:])
        ax2[i].set_xlim((0,nlam))
        ax2[i].set_title(title[i])
    fig2.savefig('profiles.png')
    
if __name__ == "__main__":
    main()