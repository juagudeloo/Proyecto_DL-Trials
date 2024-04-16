import torch
from torch import nn
import numpy as np
from muram import MuRAM
from nn_model import *
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from timeit import default_timer as timer 



def main():
    ptm = "/girg/juagudeloo/MURAM_data/Numpy_MURAM_data/"
    pth_out = "Results/"
    training_files = ["085000", "090000", "095000", "100000", "105000", "110000"]

    #Creating the model for training
    model_0 = InvModel1(300,4*20,4096).float()
    #Defining the agnostic device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nThe model will be runned in:", device)
    #Model training hyperparams
    loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.1)
    epochs = 40

    #Training

    # Set the seed and start the timer
    torch.manual_seed(42) #seed for the random weights of the model
    train_time_start_on_cpu = timer()

    train_loss_history = np.zeros((epochs,))
    test_loss_history = np.zeros((epochs,))
    test_acc_history = np.zeros((epochs,))

    for filename in training_files:
        #Creation of the muram data processing object
        muram = MuRAM(ptm = ptm, pth_out = pth_out, filename = filename)

        tr_input, test_input, tr_output, test_output = muram.train_test_sets("Stokes")

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

        BATCH_SIZE = 32

        train_dataloader = DataLoader(train_data,
            batch_size=BATCH_SIZE, # how many samples per batch? 
            shuffle=True # shuffle data every epoch?
        ).to(device)

        test_dataloader = DataLoader(test_data,
            batch_size=BATCH_SIZE,
            shuffle=False # don't necessarily have to shuffle the testing data
        ).to(device)

        print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
        print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
        train_features_batch, train_labels_batch = next(iter(train_dataloader))
        print(f"""
    Shape of each batch input and output:
    train input batch shape: {train_features_batch.shape}, 
    train output batch shape: {train_labels_batch.shape}
            """ )
        
        

        #Create model save path 
        MODEL_PATH = Path(pth_out+"model_weights/")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        MODEL_NAME = "inversion1.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

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
            train_loss_history[epoch] = train_loss

            
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
                    test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
                
                # Calculations on test metrics need to happen inside torch.inference_mode()
                # Divide total test loss by length of test dataloader (per batch)
                test_loss /= len(test_dataloader)
                test_loss_history[epoch] = test_loss


                # Divide total accuracy by length of test dataloader (per batch)
                test_acc /= len(test_dataloader)
                test_acc_history[epoch] = test_acc

            ## Print out what's happening
            print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

        # Calculate training time      
        train_time_end_on_cpu = timer()
        total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                                end=train_time_end_on_cpu,
                                                device=str(next(model_0.parameters()).device))



        # the model state dict after training
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
                f=MODEL_SAVE_PATH)
    

    

if __name__ == "__main__":
    main()