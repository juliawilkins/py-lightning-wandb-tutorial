import os
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
        output = self.model(x)
        # Can add extra layers here too
        return output
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params['LEARNING_RATE'])
        return optimizer 
    
    def training_step(self, batch, batch_idx):
        # Batch is a batch of data 
        # Batch idx is the index of the batch (useful for if you only want to log some things some batches)
        data, target = batch 
        output = self(data) # self(data) calls forward() function here
        loss = self.loss_fn(output, target)

        # Easy logging here for training step
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        # prog_bar will print the losses in progress bar
        # on_step will log to wandb called "train_loss_step" and "train_loss_epoch"
        # if only one of on_step/on_epoch is true, will log as plain "train_loss"

        # Maybe you want to plot something every 10 epochs, not always bc it's slow
        # if batch_idx == 10:
        #     plt.plot(dummyplot)
        
        # Accuracy per epoch
        preds = output.argmax(dim=1)
        acc = self.accuracy_metric(preds, target)
        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True)
        return loss 


    def validation_step(self, batch, batch_idx):
        # Model is automatically in eval mode here 
        data, target = batch 
        output = self(data) # self(data) calls forward() function here
        loss = self.loss_fn(output, target)

        # Easy logging here for training step
        # Set to only log at end of epoch
        self.log('val_loss_epoch', loss,  prog_bar=True, on_step=False, on_epoch=True)
        
        preds = output.argmax(dim=1)
        acc = self.accuracy_metric(preds, target)
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True)
        return loss 
    
    def test_step(self, batch, batch_idx):
        # Model is automatically in eval mode here 
        data, target = batch 
        output = self(data) # self(data) calls forward() function here

        preds = output.argmax(dim=1)
        acc = self.accuracy_metric(preds, target)
        self.log('test_acc', acc, on_epoch=True)
        # Easy logging here for training step
        # Set to only log at end of epoch
        # self.log('val_loss_epoch', loss,  prog_bar=True, on_step=False, on_epoch=True)
        # return loss 



def main():
    DATA_DIR = 'ESC-50-master/audio'
    CSV_PATH = 'ESC-50-master/meta/esc50.csv' 
    MODEL_CKPTS_DIR = "trained_models"
    os.makedirs(MODEL_CKPTS_DIR, exist_ok=True)
    
    # Set up parameters
    PARAMS = {
        'RANDOM_STATE': 42,
        'TEST_SIZE': 0.2,
        'VAL_SIZE': 0.25,
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.001,
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

    # Initialize the model
    model = PLSimpleCNN(PARAMS)

    # PL Lightning Magic!

    # Initialize weights and biases logger
    logger = WandbLogger(
            project="PL_WANDB_Tutorial",
        )
    # Log the parameters dictionary to weights and biases
    logger.log_hyperparams(PARAMS) 

    # Trainer object is very customizable
    # Simple version
    trainer = L.Trainer(
        logger=logger,
        max_epochs=PARAMS['N_EPOCHS'],
        devices="auto",
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=MODEL_CKPTS_DIR,
                filename="esc-classification",
                monitor="val_acc_epoch",
                auto_insert_metric_name=True,
                save_top_k=1 # save top two best models for this criteron
            ),
        ],
        )
    # Training and validation loops under the hood
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Test loop
    trainer.test(model=model, 
                 dataloaders=test_loader,
                 ckpt_path=f"{MODEL_CKPTS_DIR}/esc-classification.ckpt",
                 verbose=True)


if __name__ == '__main__':
    main()
