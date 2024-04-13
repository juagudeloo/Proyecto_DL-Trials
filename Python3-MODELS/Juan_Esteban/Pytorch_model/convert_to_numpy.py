import numpy as np

def main():
    #Function for converting every file to numpy binaries
    ptm = "/girg/juagudeloo/MURAM_data/"
    filenames = []
    for i in range(52,223+1):
        try:
            if i < 100:
                fln = "0"+str(i)+"000"
            if i >= 100:
                fln = str(i)+"000"
            filenames.append(fln)
        except:
            continue
    
    nx = 480 
    ny = 256
    nz = 480
    for filename in filenames:
        in_dir = ptm+"raw_MURAM_data/"
        out_dir = ptm+"Numpy_MuRAM_data/"
        
        #TEMPERATURE
        print("Charging temperature data...")
        #From charging the EOS data, we obtain the temperature information by taking the 0 index
        #in the 0 axis.
        mtpr = np.memmap.reshape(np.memmap(in_dir+"eos."+filename,dtype=np.float32), 
                                (2,nx,ny,nz), order="A")[0,:,:,:]
        np.save(out_dir+"mtpr_"+filename, mtpr)
        print("Temperature done!")
        
        #DENSITY
        print("Charging density...")
        mrho = np.memmap(in_dir+"result_0."+filename,dtype=np.float32)
        np.save(out_dir+"mrho_"+filename, mrho)
        del mrho
        print("Density done!")

        #VELOCITY COMPONENTS
        #We divide the charged data by the density to obtain the actual velocity from the momentum.
        print("Charging velocity components...")
        mvxx = np.memmap(in_dir+"result_1."+filename,dtype=np.float32)
        np.save(out_dir+"mvxx_"+filename, mvxx)
        del mvxx
        
        mvyy = np.memmap(in_dir+"result_2."+filename,dtype=np.float32)
        np.save(out_dir+"mvyy_"+filename, mvyy)
        del mvyy
        
        mvzz = np.memmap(in_dir+"result_3."+filename,dtype=np.float32)
        np.save(out_dir+"mvzz_"+filename, mvzz)
        del mvzz
        print("velocity components done!")
        
        
        #MAGNETIC FIELD COMPONENTS
        print("Charging magn. field components...")
        #Conversion coefficient for magnetic field to be converted in cgs units.
        coef = np.sqrt(4.0*np.pi)
        
        mbxx =  coef*np.memmap(in_dir+"result_5."+filename,dtype=np.float32)
        np.save(out_dir+"mbxx_"+filename, mbxx)
        del mbxx
        
        mbyy =  coef*np.memmap(in_dir+"result_6."+filename,dtype=np.float32)
        np.save(out_dir+"mbyy_"+filename, mbyy)
        del mbyy
        
        mbzz =  coef*np.memmap(in_dir+"result_7."+filename,dtype=np.float32)
        np.save(out_dir+"mbzz_"+filename, mbzz)
        print("Magn. field components done!")
        del mbzz

if __name__=="__main__":
    main()