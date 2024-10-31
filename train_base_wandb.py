import os
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import AudioSpectrogramDataset
from model import SimpleCNN

def train(model, device, train_loader, optimizer, criterion, epoch, train_len):
    model.train()
    train_correct = 0
    train_loss_epoch = 0
    train_accuracy_epoch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss_epoch += loss.item()
        loss.backward()
        optimizer.step()

        # Accuracy for train
        train_pred = output.argmax(dim=1, keepdim=True)
        train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
        train_accuracy = 100. * train_correct / train_len

        train_accuracy_epoch += train_accuracy
        # Log batch metrics to wandb
        wandb.log({'train_loss': loss.item(), 'epoch': epoch})

        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} Accuracy: {train_correct}/{train_len} ({train_accuracy:.0f}%)\n')
    
    return {'epoch': epoch, 'Train_loss_epoch': train_loss_epoch / train_len, 'Train_accuracy_epoch': train_accuracy_epoch / train_len}
    
def test(model, device, val_loader, criterion, epoch, val_len):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            val_loss += loss * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    average_loss = val_loss / val_len
    accuracy = 100. * correct / val_len

    # Log epoch metrics to wandb
    wandb.log({'val_loss': average_loss, 'epoch': epoch})

    print(f'\nValidation set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{val_len} ({accuracy:.0f}%)\n')
    return {'epoch': epoch, 'val_loss': average_loss, 'val_accuracy': accuracy}

def main():
    # Initialize wandb
    wandb.init(project='PL_WANDB_Tutorial', config={
        'RANDOM_STATE': 42,
        'VAL_SIZE': 0.25,
        'TEST_SIZE': 0.2,
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.0001,
        'N_EPOCHS': 10,
        'N_CLASSES': 50 
    })
    config = wandb.config

    # Load data and prepare dataset
    DATA_DIR = 'ESC-50-master/audio'
    CSV_PATH = 'ESC-50-master/meta/esc50.csv'
    OUT_DIR = 'trained_models'
    
    df = pd.read_csv(CSV_PATH)
    data = df[['filename', 'target']]
    # this gets 60/20/20 split
    train_df, test_df = train_test_split(data, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_df, val_df = train_test_split(train_df, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(config.N_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR, train_df), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR, val_df), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR, test_df), batch_size=config.BATCH_SIZE, shuffle=False)

    for epoch in range(1, config.N_EPOCHS + 1):
        train_results = train(model, device, train_loader, optimizer, criterion, epoch, len(train_df))
        val_results = test(model, device, val_loader, criterion, epoch, len(val_df))
        
        # Log results per epoch to wandb
        wandb.log(train_results)
        wandb.log(val_results)
        
        # Save the model (set up for latest epoch only)
        os.makedirs(OUT_DIR, exist_ok=True)
        MODEL_OUT_PATH = f"{OUT_DIR}/esc_classifier_ep{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'val_loss': average_loss,
            # 'val_acc': accuracy
            }, MODEL_OUT_PATH)

        # Save the model as an artifact in wandb
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(MODEL_OUT_PATH)
        wandb.log_artifact(artifact)

if __name__ == '__main__':
    main()
