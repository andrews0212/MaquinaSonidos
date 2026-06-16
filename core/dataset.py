import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class DCASEAudioDataset(Dataset):
    def __init__(self, data_dir, n_mels=128, max_frames=320):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        ruta_audio = self.file_paths[idx]
        y, sr = librosa.load(ruta_audio, sr=None)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_normalized = (mel_spec_db + 80) / 80
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
        
        if mel_spec_normalized.shape[1] < self.max_frames:
            pad_width = self.max_frames - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_normalized = mel_spec_normalized[:, :self.max_frames]
            
        return torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0)