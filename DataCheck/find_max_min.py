import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():
    quant_filenames = ["mtpr", "mrho", "mbxx", "mbyy", "mbzz", "mvxx", "mvyy", "mvzz"]
    step = 1000
    n_files = np.arange(80000, 200000+step, step)
    n_files = np.delete(n_files, np.argwhere(n_files == 98000))
    n_filenames = n_files.copy().astype(str)
    for i in range(len(n_files)):
        n_filenames[i] = "0"+n_filenames[i] if n_files[i] < 100000 else n_filenames[i]
        
    info_dict = {"index": ["T", "Rho", "Bxx", "Byy", "Bzz", "Vxx", "Vyy", "Vzz"], 
                 "max_mean": [], 
                 "max_std": [], 
                 "min_mean": [],
                 "min_std": []
                 }
    
    data_path = "../../MURAM_data/Numpy_MURAM_data/"
    for quant in quant_filenames:
        print(quant)
        max_values = []
        min_values = []
        for n in n_filenames:
            if quant in ["mvxx", "mvyy", "mvzz"]:
                if int(n) % 10000 == 0:
                    print(n)
                quant_npy = np.load(data_path+quant+"_"+n+".npy")
                mrho = np.load(data_path+"mrho"+"_"+n+".npy")
                quant_npy = quant_npy/mrho
                max_values.append(np.max(quant_npy))
                min_values.append(np.min(quant_npy))
            else:
                if int(n) % 10000 == 0:
                    print(n)
                quant_npy = np.load(data_path+quant+"_"+n+".npy")
                max_values.append(np.max(quant_npy))
                min_values.append(np.min(quant_npy))
        
        max_mean = np.mean(max_values)
        max_std = np.std(max_values)
        min_mean = np.mean(min_values)
        min_std = np.std(min_values)
        
        info_dict["max_mean"].append(max_mean)
        info_dict["max_std"].append(max_std)
        info_dict["min_mean"].append(min_mean)
        info_dict["min_std"].append(min_std)
        
    
    
    
    
    df = pd.DataFrame(info_dict)
    df.to_csv("max_min_values.csv")
    
    
if __name__ == "__main__":
    main()