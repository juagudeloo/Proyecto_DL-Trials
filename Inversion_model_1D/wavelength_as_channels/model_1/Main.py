#This new main saves all the files data in the same TestDataset.
import torch
from torch import nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from timeit import default_timer as timer 
import os
import datetime
import time
import sys

sys.path.append("/girg/juagudeloo/Proyecto_DL-Trials/Inversion_model_1D/module")
from muram import MuRAM
from nn_model import *


def main():
    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    training_files = ["085000", 
    "090000","095000", "100000", "105000", "110000"
    ]

    #Creating the model for training
    model = InvModel1(36,6*20,4096).float()

    #Defining the agnostic device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nThe model will be runned in:", device)

    #Model training hyperparams
    model.to(device)
    loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places
    lr = 5e-4
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    epochs = 12
    
    #Training
    results_out = "Results/"
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

    # Set the seed and start the timer
    torch.manual_seed(42) #seed for the random weights of the model
    train_time_start_on_cpu = timer()

    train_loss_history = np.zeros((epochs,))
    test_loss_history = np.zeros((epochs,))
    test_acc_history = np.zeros((epochs,))
    	
    start = time.time()
    
    #Validation dataset
    val_muram = MuRAM(ptm = ptm, filenames = [''])
    val_atm_quant, val_stokes = val_muram.charge_quantities(filename = "130000", vertical_comp = False)
    val_atm_quant_tensor = torch.from_numpy(val_atm_quant).to(device)
    val_atm_quant_tensor = torch.reshape(val_atm_quant_tensor, (480*480,20,6))
    stokes_tensor = torch.from_numpy(val_stokes).to(device)
    stokes_tensor = torch.reshape(stokes_tensor, (480*480,stokes_tensor.size()[2],stokes_tensor.size()[3]))

    BATCH_SIZE = 80
    print(stokes_tensor.size(), val_atm_quant_tensor.size())
    validation_data = TensorDataset(stokes_tensor, val_atm_quant_tensor)
    validation_dataloader = DataLoader(validation_data,
            batch_size=BATCH_SIZE,
            shuffle=False # don't necessarily have to shuffle the testing data
        )

    #Creation of the muram data processing object
    muram = MuRAM(ptm = ptm, filenames = training_files)

    train_data, test_data= muram.train_test_sets(vertical_comp = False)

    BATCH_SIZE = 80

    train_dataloader = DataLoader(train_data,
        batch_size=BATCH_SIZE, # how many samples per batch? 
        shuffle=True # shuffle data every epoch?
    )

    test_dataloader = DataLoader(test_data,
        batch_size=BATCH_SIZE,
        shuffle=False # don't necessarily have to shuffle the testing data
    )    

    print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"""
    Shape of each batch input and output:
    train input batch shape: {train_features_batch.shape}, 
    train output batch shape: {train_labels_batch.shape}
        """ )

    #Charge the weights in case there have been some training before
    if MODEL_SAVE_PATH.exists():
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


    #############################################################################
    # TRAINING
    #############################################################################
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
        validation_step(pth_out, model, validation_dataloader, val_atm_quant, muram.nx, muram.nz, epoch)
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
        
    np.save(metrics_out+"train_loss_history"+str(epochs)+"E"+str(lr)+"lr"+".npy", train_loss_history)
    np.save(metrics_out+"test_loss_history"+str(epochs)+"E"+str(lr)+"lr"+".npy", test_loss_history)
    np.save(metrics_out+"test_acc_history"+str(epochs)+"E"+str(lr)+"lr"+".npy", test_acc_history)
    runtime = time.time()-start
    with open(metrics_out+"runtime.txt", "w") as f:
	    f.write(str(datetime.timedelta(seconds=runtime)))
    
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

def validation_step(pth_out, model, validation_dataloader, val_atm_quant, nx, nz, epoch):
    validated_atm = torch.zeros((480*480,80))
    with torch.inference_mode():
        i = 0
        for X, y in validation_dataloader:
            # 1. Forward pass
            valid_pred = model.double()(X.double())
            validated_atm[i*80:(i+1)*80] = valid_pred
            i += 1
        validated_atm = torch.reshape(validated_atm, (nx, nz, 20, 4))
        validated_atm = validated_atm.to("cpu").numpy()
            
        print("Validation done!")
    
    #Making a plot of the correlation plots of the validation set.
    titles = ["T", "rho", "By", "vy"]
    
    validation_visual(validated_atm, val_atm_quant, epoch_to_plot=f"epoch {epoch+1}", images_out=pth_out, titles=titles)

if __name__ == "__main__":
    main()
