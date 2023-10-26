import numpy as np
import train_generate.model_prof_tools as mpt
def main():
    ptm = "/girg/juagudeloo/MURAM_data/"
    filename = "085000"
    
    global idl, irec, f # Save values between calls
    filename = filename
    [int4f,intf,flf]=mpt.check_types()
    stk_filename = filename+"_0000_0000.prof"
    profs = [] #It's for the reshaped data - better for visualization.
    profs_ravel = [] #its for the ravel data to make the splitting easier.
    #Charging the stokes profiles for the specific file
    print(f"reading Stokes params {stk_filename}")

    nx = 480, 
    ny = 256, 
    nz = 480
    file_type = "nicole"
    nlam = 300 
    N_profs = 4


    
    for ix in range(nx):
        for iy in range(nz):
            p_prof = mpt.read_prof(ptm+stk_filename, file_type,  nx, nz, nlam, iy, ix)
            p_prof = np.memmap.reshape(np.array(p_prof), (nlam, N_profs))
            ##############################################################################
            #profs_ravel is going to safe all the data in a one dimensional array where
            #the dimensional indexes are disposed as ix*nz+iy.
            ##############################################################################
            profs.append(p_prof)  
    print("scaling...")
    profs = np.array(profs) #this step is done so that the array has the same shape as the ouputs referring to the four type of data it has
    
    profs = np.memmap.reshape(profs,(nx, nz, nlam, N_profs))


    profs_fln = filename+"_prof.npy"
    np.save(profs_fln, profs)

    new_profs = np.load(profs_fln)
    print("max:", np.max(new_profs), "min:", np.min(new_profs))

    print(f"Stokes params done! {filename}")


if __name__ == "__main__":
    main()