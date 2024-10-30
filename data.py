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

        waveform, curr_sr = torchaudio.load(os.path.join(self.data_dir, audio_path))
        waveform = torchaudio.functional.resample(waveform, curr_sr, self.sr)
        spectrogram = self.to_mel_spec(waveform, self.sr)  # Adjusted to use actual sample rate

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label