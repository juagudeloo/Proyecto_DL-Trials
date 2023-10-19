import numpy as np

def main():
    ptm = "/girg/juagudeloo/MURAM_data/"
    filename = "085000"
    stokes = np.memmap(ptm+filename+"_prof.npy", dtype = np.float32)
    print("min:", np.min(stokes), "max:", np.max(stokes))


if __name__ == "__main__":
    main()