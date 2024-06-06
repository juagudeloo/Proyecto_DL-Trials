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

sys.path.append("/girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/Pytorch_model/Inversion/1D/module")
from muram import MuRAM
from nn_model import *


def main():
    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    training_files = ["085000", "090000", 
      #                "095000", "100000", "105000", "110000"
    ]

    #Creating the model for training
    model_0 = InvModel1(4,4*20,4096).float()

    #Defining the agnostic device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nThe model will be runned in:", device)
    #Model training hyperparams
    model_0.to(device)
    loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places
    lr = 1e-4
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=lr)
    epochs = 3
    results_out = "Results/"
    if not os.path.exists(results_out):
        os.mkdir(results_out)
    pth_out = results_out+f"{epochs}E_"+f"{lr}lr/"
    if not os.path.exists(pth_out):
        os.mkdir(pth_out)
    
    #Create model save path 
    MODEL_PATH = Path(pth_out+"model_weights/")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "inversion_"+f"{epochs}E_"+f"{lr}lr.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    

    # Set the seed and start the timer
    torch.manual_seed(42) #seed for the random weights of the model
    train_time_start_on_cpu = timer()

    train_loss_history = np.zeros((epochs*len(training_files),))
    test_loss_history = np.zeros((epochs*len(training_files),))
    test_acc_history = np.zeros((epochs*len(training_files),))

    start = time.time()
    ifl = 0
    
    #Validation dataset
    val_muram = MuRAM(ptm = ptm, filename = "130000")
    val_atm_quant, val_stokes = val_muram.charge_quantities()
    val_atm_quant_tensor = torch.from_numpy(val_atm_quant).to(device)
    val_atm_quant_tensor = torch.reshape(val_atm_quant_tensor, (480*480,20,4))
    stokes_tensor = torch.from_numpy(val_stokes).to(device)
    stokes_tensor = torch.reshape(stokes_tensor, (480*480,300,4))
    stokes_tensor = torch.moveaxis(stokes_tensor,1,2)
    # stokes = torch.reshape(stokes,(stokes_s[0]*stokes_s[1], stokes_s[2], stokes_s[3]))
    BATCH_SIZE = 80

    validation_data = TensorDataset(stokes_tensor, val_atm_quant_tensor)
    validation_dataloader = DataLoader(validation_data,
            batch_size=BATCH_SIZE,
            shuffle=False # don't necessarily have to shuffle the testing data
        )

    #Validation lists
    val_atm_list = []
    epochs_to_plot = []
    
    for filename in training_files:
        #Creation of the muram data processing object
        muram = MuRAM(ptm = ptm, filename = filename)

        tr_input, test_input, tr_output, test_output = muram.train_test_sets("Stokes")
        tr_input = torch.moveaxis(tr_input, 1,2)
        test_input = torch.moveaxis(test_input, 1,2)
        #Testing wavelength as channels and the output as a fully connected layer
        tr_output = torch.reshape(tr_output, (tr_output.size()[0], tr_output.size()[1]*tr_output.size()[2]))
        test_output = torch.reshape(test_output, (test_output.size()[0], test_output.size()[1]*test_output.size()[2]))

        print(f"""
    Shape of the data
            tr_input shape ={tr_input.size()}
            test_input shape = {test_input.size()}
            tr_output shape = {tr_output.size()}
            test_output shape = {test_output.size()}
            """)
        
        #Train and test dataloader
        train_data = TensorDataset(tr_input.to(device), tr_output.to(device))
        test_data = TensorDataset(test_input.to(device), test_output.to(device))

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
            model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

        
        
        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n-------")
            ### Training
            train_loss = 0
            # Add a loop to loop through training batches
            for batch, (X, y) in enumerate(train_dataloader):
                model_0.train() 
                # 1. Forward pass
                X, y = X.to(device), y.to(device)
                y_pred = model_0.double()(X.double())

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
            train_loss_history[ifl*epochs+epoch] = train_loss

            
            ### Testing
            # Setup variables for accumulatively adding up loss and accuracy 
            test_loss, test_acc = 0, 0 
            model_0.eval()
            with torch.inference_mode():
                for X, y in test_dataloader:
                    # 1. Forward pass
                    test_pred = model_0(X)
                
                    # 2. Calculate loss (accumatively)
                    test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

                    # 3. Calculate accuracy (preds need to be same as y_true)
                    test_acc += accuracy_fn(y_true=y.argmax(dim=1), y_pred=test_pred.argmax(dim=1))
                
                # Calculations on test metrics need to happen inside torch.inference_mode()
                # Divide total test loss by length of test dataloader (per batch)
                test_loss /= len(test_dataloader)
                test_loss_history[ifl*epochs+epoch] = test_loss


                # Divide total accuracy by length of test dataloader (per batch)
                test_acc /= len(test_dataloader)
                test_acc_history[ifl*epochs+epoch] = test_acc

            ## Print out what's happening
            print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

            if (epoch % 3 == 0) or (epoch == epochs-1):
                print("\nValidation plot...")
                #validation plot
                validated_atm = torch.zeros((480*480,80))
                with torch.inference_mode():
                    i = 0
                    for X, y in validation_dataloader:
                        # 1. Forward pass
                        valid_pred = model_0.double()(X.double())
                        validated_atm[i*80:(i+1)*80] = valid_pred
                        i += 1
                    val_atm_list.append(np.reshape(validated_atm, (muram.nx, muram.nz, 20, 4)))
                    epochs_to_plot.append(filename+f" epoch {epoch+1}")
                
            print("Validation done!")
    
        #Making an animation of the correlation plots of the validation set.
        titles = ["T", "rho", "By", "vy"]
        validation_visual(val_atm_list, val_atm_quant, epochs_to_plot, pth_out, titles)
            
        # Calculate training time      
        train_time_end_on_cpu = timer()
        total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                                end=train_time_end_on_cpu,
                                                device=str(next(model_0.parameters()).device))



        # the model state dict after training
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
                f=MODEL_SAVE_PATH)
        ifl += 1
        
    metrics_out = pth_out+"loss_metrics/"
    if not os.path.exists(metrics_out):
        os.mkdir(metrics_out)
	
    np.save(metrics_out+"train_loss_history"+str(epochs)+"E_"+str(lr)+"lr"+".npy", train_loss_history)
    np.save(metrics_out+"test_loss_history"+str(epochs)+"E_"+str(lr)+"lr"+".npy", test_loss_history)
    np.save(metrics_out+"test_acc_history"+str(epochs)+"E_"+str(lr)+"lr"+".npy", test_acc_history)
    runtime = time.time()-start
    with open(metrics_out+"runtime.txt", "w") as f:
	    f.write(str(datetime.timedelta(seconds=runtime)))
    

    

if __name__ == "__main__":
    main()
