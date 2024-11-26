import os
from pathlib import Path
import sys

import time
from timeit import default_timer as timer 

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from tqdm import tqdm

#Own modules
sys.path.append("/girg/juagudeloo/Proyecto_DL-Trials/Inversion_model_1D/module")
from utils.train_utils  import *


def train_model(
                ptm: str,
                model: nn.Module,
                lr: float,
                epochs: int,
                batch_size: int,
                vertical_comp: bool,
                opt_depth_stratif: bool
                ) -> None:
    
    """
    Function to train the model.
    -----------------------------
    """
    
    
    #Path to the data
    training_files = ["085000", 
    "090000","095000", "100000", "105000", "110000"
    ]

    #Model training hyperparams
    loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    #Defining the agnostic device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("\nThe model will be runned in:", device)
    
    # Create the model save path
    MODEL_SAVE_PATH, pth_out = create_model_save_path(epochs, lr)
    #Charge the weights in case there have been some training before
    if MODEL_SAVE_PATH.exists():
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    
    # Set timers
    train_time_start_on_cpu = timer()
    start = time.time()
    
    #Data loaders
    train_dataloader, test_dataloader, nx, nz  = train_test_dl(ptm, training_files, vertical_comp, batch_size, opt_depth_stratif=opt_depth_stratif)
    
    validation_dataloader, val_atm_quant = validation_dl(device, ptm, vertical_comp, batch_size, opt_depth_stratif=opt_depth_stratif)
    
    #############################################################################
    # TRAINING
    #############################################################################
    train_loss_history = np.zeros((epochs,))
    test_loss_history = np.zeros((epochs,))
    test_acc_history = np.zeros((epochs,))
    
    total_train_time_model = 0
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")
        ### Training
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        train_loss_history[epoch] = train_loss
        
        ### Testing
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn)
        test_loss_history[epoch] = test_loss
        test_acc_history[epoch] = test_acc

        ## Print out what's happening
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

        print("\nValidation plot...")
        #validation plot
        validation_step(pth_out, model, validation_dataloader, val_atm_quant, nx, nz, epoch, vertical_comp)
        print("\nplotted...")
        

        # Calculate training time      
        train_time_end_on_cpu = timer()
        total_train_time_model += print_train_time(start=train_time_start_on_cpu, 
                                                end=train_time_end_on_cpu,
                                                device=str(next(model.parameters()).device))

        # the model state dict after training
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
                f=MODEL_SAVE_PATH)
        
    metrics_out = pth_out+"loss_metrics/"
    if not os.path.exists(metrics_out):
        os.mkdir(metrics_out)
        
    train_loss_history_path = metrics_out+"train_loss_history"+str(epochs)+"E"+str(lr)+"lr"+".npy"
    test_loss_history_path = metrics_out+"test_loss_history"+str(epochs)+"E"+str(lr)+"lr"+".npy"
    test_acc_history_path = metrics_out+"test_acc_history"+str(epochs)+"E"+str(lr)+"lr"+".npy"
    
    np.save(train_loss_history_path, train_loss_history)
    np.save(test_loss_history_path, test_loss_history)
    np.save(test_acc_history_path, test_acc_history)
    
    runtime = time.time()-start
    with open(metrics_out+"runtime.txt", "w") as f:
        f.write(str(datetime.timedelta(seconds=runtime)))
     
    return train_loss_history_path, test_loss_history_path, test_acc_history_path
