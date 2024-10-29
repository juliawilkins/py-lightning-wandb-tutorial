import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data import AudioSpectrogramDataset
from model import SimpleCNN

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_results = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Collecting data for logging using a dictionary
        train_results.append({'epoch': epoch, 'batch': batch_idx, 'loss': round(loss.item(), 4)})
        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return train_results

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0  # To count total number of samples tested

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            test_loss += loss * data.size(0)  # Multiply by batch size for accurate mean calculation later
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    average_loss = test_loss / total
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')

    # Prepare the test results summary
    test_results_summary = {
        'epoch': epoch,
        'correct_predictions': correct,
        'average_loss': round(average_loss, 4)
    }
    
    return test_results_summary
    

def main():
    # Load CSV file and prepare the dataset
    DATA_DIR = 'ESC-50-master/audio'
    CSV_PATH = 'ESC-50-master/meta/esc50.csv' 
    TRAIN_LOG_PATH = 'train_results.csv'
    TEST_LOG_PATH = 'test_results.csv'
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    N_EPOCHS = 2
    
    df = pd.read_csv(CSV_PATH)

    data = df[['filename', 'target']]

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR, train_df), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(AudioSpectrogramDataset(DATA_DIR, test_df), batch_size=BATCH_SIZE, shuffle=False)

    all_train_results = []
    all_test_results = []
    
    for epoch in range(1, N_EPOCHS+1):
        epoch_train_results = train(model, device, train_loader, optimizer, criterion, epoch)
        all_train_results.extend(epoch_train_results)
        
        epoch_test_summary = test(model, device, test_loader, criterion, epoch)
        all_test_results.append(epoch_test_summary)

    # Convert results to DataFrame and write to CSV
    pd.DataFrame(all_train_results).to_csv(TRAIN_LOG_PATH, index=False)
    pd.DataFrame(all_test_results).to_csv(TEST_LOG_PATH, index=False)

if __name__ == '__main__':
    main()