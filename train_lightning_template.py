import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from data import AudioSpectrogramDataset
from model import SimpleCNN

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from data import AudioSpectrogramDataset
from model import SimpleCNN

class PLSimpleCNN(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # Initialize model from other file
        self.model = SimpleCNN(params)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=params["N_CLASSES"])

    def forward(self, x):
        # TODO: Define forward pass
        return 

    def configure_optimizers(self):
        # TODO: Configure the optimizer        
        return 

    def training_step(self, batch, batch_idx):
        # TODO: Define a single training step:
        # - Extract data and target from the batch
        # - Pass data through the forward() method
        # - Calculate loss and log it
        # - Calculate and log accuracy
        
        return 

    def validation_step(self, batch, batch_idx):
        # TODO: Define a single validation step
        # - Extract data and target from the batch
        # - Pass data through the model to get the output
        # - Calculate and log the validation loss and accuracy
        
        return 

    def test_step(self, batch, batch_idx):
        # TODO: Define a single test step (this is the same as validation)
        # - Extract data and target from the batch
        # - Pass data through the model to get the output
        # - Calculate and log accuracy
        
        return 




def main():
    DATA_DIR = 'ESC-50-master/audio'
    CSV_PATH = 'ESC-50-master/meta/esc50.csv' 
    MODEL_OUT_DIR = "trained_models"
    
    # Set up parameters
    PARAMS = {
        'RANDOM_STATE': 42,
        'TEST_SIZE': 0.2,
        'VAL_SIZE': 0.25,
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.01,
        'N_EPOCHS': 100,
        'NUM_GPUS': 1 if torch.cuda.is_available() else "cpu",
        'N_CLASSES': 50,
        'N_WORKERS': 2
        }
    
    # Set up dataset
    # Load CSV file and prepare the dataset

    df = pd.read_csv(CSV_PATH)
    data = df[['filename', 'target']]

    # Split the data into training and testing sets (60/20/20)
    train_df, test_df = train_test_split(data, test_size=PARAMS['TEST_SIZE'],
                                          random_state=PARAMS['RANDOM_STATE'])
    train_df, val_df = train_test_split(train_df, test_size=PARAMS['VAL_SIZE'],
                                          random_state=PARAMS['RANDOM_STATE'])

    # Initialize dataloader
    train_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR, train_df),
                               batch_size=PARAMS['BATCH_SIZE'], 
                               shuffle=True, num_workers=PARAMS['N_WORKERS'],
                               persistent_workers=True)
    val_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR,val_df),
                              batch_size=PARAMS['BATCH_SIZE'], 
                              shuffle=False, num_workers=PARAMS['N_WORKERS'],
                              persistent_workers=True)
    test_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR,test_df),
                              batch_size=PARAMS['BATCH_SIZE'], 
                              shuffle=False, num_workers=PARAMS['N_WORKERS'],
                              persistent_workers=True)


    # PL Lightning Magic!
    # Initialize the model
    model = PLSimpleCNN(PARAMS)

    # Initialize weights and biases logger
    # TODO 

    # Log the parameters dictionary to weights and biases
    # TODO 

    # Define your Trainer() object
    # TODO

    # Run the trainer, using trainer.fit(..customize_these_args)
    # TODO 

    # Optionally call trainer.test(...args) for final evaluation
    # TODO
    


if __name__ == '__main__':
    main()
