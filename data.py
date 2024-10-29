import os

from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as Ta

class AudioSpectrogramDataset(Dataset):
    def __init__(self, data_dir, dataframe, transform=None):
        self.data_dir = data_dir
        self.audio_paths = dataframe['filename'].tolist()
        self.labels = dataframe['target'].tolist()
        self.transform = transform
    
    def toMelSpectrogram(self, waveform, sample_rate):
        spectrogram = Ta.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                hop_length=239,
                n_mels=50,
                normalized=True
            )(waveform)
        return spectrogram

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(os.path.join(self.data_dir, audio_path))
        spectrogram = self.toMelSpectrogram(waveform, sample_rate)  # Adjusted to use actual sample rate

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label