import os
import sys
from pathlib import Path

import datetime

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

import torch
from torch.utils.data import TensorDataset, DataLoader

#Own modules
sys.path.append("/girg/juagudeloo/Proyecto_DL-Trials/Inversion_model_1D/module")
from muram import MuRAM

def create_model_save_path(epochs: int, lr: float, results_out: str = "Results/"):
    #Training
    if not os.path.exists(results_out):
        os.mkdir(results_out)
    pth_out = results_out+f"{epochs}E_"+f"{lr}lr/"
    if not os.path.exists(pth_out):
        os.mkdir(pth_out)
    #Create model save path
    MODEL_PATH = Path(pth_out+"model_weights/")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "inversion_"+str(epochs)+"E"+str(lr)+"lr"+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    return MODEL_SAVE_PATH

def train_test_dl(ptm: str, training_files: list, vertical_comp: bool, batch_size: int):
    # Set the seed and start the timer
    torch.manual_seed(42) #seed for the random weights of the model

    #Creation of the muram data processing object
    muram = MuRAM(ptm = ptm, filenames = training_files)

    train_data, test_data= muram.train_test_sets(vertical_comp = vertical_comp)

    train_dataloader = DataLoader(train_data,
        batch_size=batch_size, # how many samples per batch? 
        shuffle=True # shuffle data every epoch?
    )

    test_dataloader = DataLoader(test_data,
        batch_size=batch_size,
        shuffle=False # don't necessarily have to shuffle the testing data
    )    
    
    print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"""
    Shape of each batch input and output:
    train input batch shape: {train_features_batch.shape}, 
    train output batch shape: {train_labels_batch.shape}
        """ )

    return train_dataloader, test_dataloader, muram.nx, muram.nz

def validation_dl(device: str, ptm: str, vertical_comp: bool, batch_size: int):

    #Validation dataset
    
    val_muram = MuRAM(ptm = ptm, filenames = [''])
    val_atm_quant, val_stokes = val_muram.charge_quantities(filename = "130000", vertical_comp = vertical_comp)
    val_atm_quant_tensor = torch.from_numpy(val_atm_quant).to(device)
    val_atm_quant_tensor = torch.reshape(val_atm_quant_tensor, (480*480,20,6))
    stokes_tensor = torch.from_numpy(val_stokes).to(device)
    stokes_tensor = torch.reshape(stokes_tensor, (480*480,stokes_tensor.size()[2],stokes_tensor.size()[3]))

    print(stokes_tensor.size(), val_atm_quant_tensor.size())
    validation_data = TensorDataset(stokes_tensor, val_atm_quant_tensor)
    validation_dataloader = DataLoader(validation_data,
            batch_size=batch_size,
            shuffle=False # don't necessarily have to shuffle the testing data
        )
    
    return validation_dataloader, val_atm_quant
    
def train_step(model, train_dataloader, loss_fn, optimizer, device):
    # Add a loop to loop through training batches
    train_loss = 0

    for batch, (X, y) in enumerate(train_dataloader):
        model.train() 
        # 1. Forward pass
        X, y = X.to(device), y.to(device)
        y_pred = model.double()(X.double())

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    return train_loss

def test_step(model, test_dataloader, loss_fn):
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss, test_acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model(X)
        
            # 2. Calculate loss (accumatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y.argmax(dim=1), y_pred=test_pred.argmax(dim=1))
        
        # Calculations on test metrics need to happen inside torch.inference_mode()

        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)
    
    return test_loss, test_acc

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def validation_step(pth_out, model, validation_dataloader, val_atm_quant, nx, nz, epoch, vertical_comp):
    validated_atm = torch.zeros((480*480,120))
    with torch.inference_mode():
        i = 0
        for X, y in validation_dataloader:
            # 1. Forward pass
            valid_pred = model.double()(X.double())
            validated_atm[i*80:(i+1)*80] = valid_pred
            i += 1
        validated_atm = torch.reshape(validated_atm, (nx, nz, 20, 6))
        validated_atm = validated_atm.to("cpu").numpy()
            
        print("Validation done!")
    
    #Making a plot of the correlation plots of the validation set.
    if vertical_comp:
        titles = ["T", "rho", "By", "vy"]
    else:
        titles = ["T", "rho", "Bqq", "Buu", "Bvv", "vy"]
    
    validation_visual(validated_atm, val_atm_quant, epoch_to_plot=f"epoch {epoch+1}", pth_out=pth_out, titles=titles)

def validation_visual(generated_quant:list, ref_quant:np.ndarray, epoch_to_plot:list, pth_out:str, titles:list):
    """
    Function for making the correlation plots.
    ------------------------------------------------
    gen_quant_list (list): list of the generated params for the specified epochs.
    ref_quant (np.ndarray): reference cube atmosphere params.
    epochs_to_plot (list): list with the name of the filename along with the epoch of training.
    images_out (str): path to save the animation.
    title (list): list of the titles corresponding to the plotted magnitudes.
    """
    
    images_out = pth_out+"validation/"
    if not os.path.exists(images_out):
        os.makedirs(images_out)
    
    N_plots = ref_quant.shape[-1]
    heights_index = [11, 8, 5, 2]
    N_heights = len(heights_index)            
    

    fig, ax = plt.subplots(N_heights, N_plots, figsize=(4*N_plots, 4*N_heights))
    tau = np.linspace(-3, 1,20)
    
    print("generated_quant.shape", generated_quant.shape)
    print("ref_quant.shape", ref_quant.shape)

    for it in range(N_heights):
        for iatm in range(N_plots):
            ax[it,iatm].scatter(generated_quant[:,:,heights_index[it],iatm].flatten(),
                            ref_quant[:,heights_index[it],:,iatm].flatten(),
                            s=5, c="darkviolet", alpha=0.1)
            
            max_x = np.max(generated_quant[:,heights_index[it],:,iatm].flatten())
            min_x = np.min(generated_quant[:,heights_index[it],:,iatm].flatten())

            max_y = np.max(ref_quant[:,:,heights_index[it],iatm].flatten())
            min_y = np.min(ref_quant[:,:,heights_index[it],iatm].flatten())

            pearson = pearsonr(generated_quant[:,:,heights_index[it],iatm].flatten(), ref_quant[:,heights_index[it],:,iatm].flatten())[0]
            ax[it,iatm].plot(generated_quant[:,heights_index[it],:,iatm],
                         generated_quant[:,heights_index[it],:,iatm],
                         "k")
            ax[it,iatm].set_title(f"{titles[iatm]} OD_{tau[heights_index[it]]:.2f} {epoch_to_plot} p_{pearson:.2f}")
            ax[it,iatm].set_xlabel("generated")
            ax[it,iatm].set_ylabel("reference")
            ax[it,iatm].set_ylim(min_y, max_y)
            ax[it,iatm].set_xlim(min_x, max_x)
    fig.tight_layout()
    fig.text(0.5, -0.02, 'Generated', ha='center',fontsize=14)
    fig.text(-0.02, 0.5, 'Original', va='center', rotation='vertical',fontsize=14)
    fig.savefig(images_out+f"visualization_{epoch_to_plot}.png")

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
