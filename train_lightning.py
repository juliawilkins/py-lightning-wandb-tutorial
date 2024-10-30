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

from data import AudioSpectrogramDataset


class PLSimpleCNN(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # Define model architecture  in init
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input shape: [batch_size, 1, 128, 1103]
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output shape: [batch_size, 16, 64, 551]

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output shape: [batch_size, 32, 32, 275]
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output shape: [batch_size, 32, 16, 137] -- Approximation

        # Update the input size of fc1 to match the flattened output size: 32 channels * 12 height * 230 width
        self.fc1 = nn.Linear(88320, 100)  # Corrected input features to match actual output
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(100, params['N_CLASSES'])  # Output classes 50 for ESC50

        # Loss function definition
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x
    
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
        
        return loss 


    def validation_step(self, batch, batch_idx):
        # Model is automatically in eval mode here 
        data, target = batch 
        output = self(data) # self(data) calls forward() function here
        loss = self.loss_fn(output, target)

        # Easy logging here for training step
        # Set to only log at end of epoch
        self.log('val_loss_epoch', loss,  prog_bar=True, on_step=False, on_epoch=True)
        return loss 



def main():
    DATA_DIR = 'ESC-50-master/audio'
    CSV_PATH = 'ESC-50-master/meta/esc50.csv' 
    MODEL_OUT_PATH = "trained_models/esc_classifier.pt"
    
    # Set up parameters
    PARAMS = {
        'RANDOM_STATE': 42,
        'TEST_SIZE': 0.2,
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.01,
        'N_EPOCHS': 2,
        'NUM_GPUS': 1 if torch.cuda.is_available() else "cpu",
        'N_CLASSES': 50,
        'N_WORKERS': 2
        }
    
    # Set up dataset
    # Load CSV file and prepare the dataset

    df = pd.read_csv(CSV_PATH)
    data = df[['filename', 'target']]

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(data, test_size=PARAMS['TEST_SIZE'],
                                          random_state=PARAMS['RANDOM_STATE'])
    train_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR, train_df),
                               batch_size=PARAMS['BATCH_SIZE'], 
                               shuffle=True, num_workers=PARAMS['N_WORKERS'])
    test_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR,test_df),
                              batch_size=PARAMS['BATCH_SIZE'], 
                              shuffle=False, num_workers=PARAMS['N_WORKERS'])

    # Initialize the model
    model = PLSimpleCNN(PARAMS)

    # PL Lightning Magic!

    # Initialize weights and biases logger
    logger = WandbLogger(
            project="esc50-classifier",
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
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )



if __name__ == '__main__':
    main()

    


#  trainer = L.Trainer(
#         logger=logger,
#         callbacks=[
#             ModelCheckpoint(
#                 dirpath=MODEL_OUT_PATH,
#                 monitor="val_loss",
#                 auto_insert_metric_name=True,
#                 save_top_k=2, # save top two best models for this criteron
#             ),
#         ],
#         devices=PARAMS["num_gpus"],
#         accelerator="gpu" if params["num_gpus"] > 0 else "cpu",
#         max_epochs=params["max_epochs"],
#         log_every_n_steps=1,
#         # limit_val_batches=0,
#         val_check_interval=0.5,
#         precision="bf16-mixed",
#         # ckpt_path="multi-disentanglement-specvq-videovq/e1sfaxy4/checkpoints/epoch=177-step=9968.ckpt",
#     )

#     # More customized version with callbacks
#     trainer = L.Trainer(
#         logger=logger,
#         callbacks=[
#             EarlyStopping(
#                 monitor="val_loss_total_step_epoch",
#                 patience=params["early_stopping_patience"],
#                 verbose=True,
#             ),
#             ModelCheckpoint(
#                 dirpath=save_model_dir,
#                 filename="{epoch}-{step}-{train_loss_total_step:.3f}-{val_loss_total_step:.3f}",
#                 monitor="val_loss_total_step",
#                 auto_insert_metric_name=True,
#                 save_top_k=5,
#             ),
#         ],
#         devices=params["num_gpus"],
#         strategy=(
#             # DDPStrategy(find_unused_parameters=False)
#             "ddp"
#             if params["num_gpus"] > 1
#             else "auto"
#         ),
#         accelerator="gpu" if params["num_gpus"] > 0 else "cpu",
#         max_epochs=params["max_epochs"],
#         log_every_n_steps=1,
#         # limit_val_batches=0,
#         val_check_interval=0.5,
#         precision="bf16-mixed",
#         # ckpt_path="multi-disentanglement-specvq-videovq/e1sfaxy4/checkpoints/epoch=177-step=9968.ckpt",
#     )
