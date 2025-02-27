import os

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as Ta
import pandas as pd

class AudioSpectrogramDataset(Dataset):
    def __init__(self, data_dir, data_split, transform=None):
        self.data_dir = data_dir
        dataframe = pd.read_csv(f'{self.data_dir}/meta/esc50.csv')
        
        if data_split == 'train':
            dataframe = dataframe.loc[dataframe['fold'] in [1,2,3]]
        elif data_split == 'val':
            dataframe = dataframe.loc[dataframe['fold'] == 4]
        if data_split == 'test':
            dataframe = dataframe.loc[dataframe['fold'] == 5]

        self.audio_paths = dataframe['filename'].tolist()
        self.labels = dataframe['target'].tolist()
        self.transform = transform
        self.sr = 16000
    
    def to_mel_spec(self, waveform, sample_rate):
        spectrogram = Ta.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                normalized=True
            )(waveform)
        return spectrogram

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        waveform, curr_sr = torchaudio.load(os.path.join(self.data_dir, f'audio/{audio_path}'))
        waveform = torchaudio.functional.resample(waveform, curr_sr, self.sr)
        spectrogram = self.to_mel_spec(waveform, self.sr)  # Adjusted to use actual sample rate.,.
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram, label
    

# val_dataset = AudioSpectrogramDataset(data_dir='ESC-50-master', data_split='val')
# val_dataloader = DataLoader(val_dataset, 
#                             batch_size=16,
#                             shuffle=False)


# for spec, label in val_dataloader:
#     import ipdb;ipdb.set_trace()
#     print(spec.shape)