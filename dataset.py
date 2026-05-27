import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DCASEAudioDataset(Dataset):
    def __init__(self, data_dir, n_mels=128, max_frames=320):
        """
        data_dir: Ruta a la carpeta 'train' (ej. dev_data_pump/train)
        n_mels: Altura de la imagen (bandas de frecuencia)
        max_frames: Anchura de la imagen (tiempo). 320 frames equivalen a unos 10 segundos de audio.
        """
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.max_frames = max_frames
        
        # Leemos todos los archivos .wav que hay en la carpeta
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
        
        print(f"Detectados {len(self.file_paths)} audios para entrenar.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Cargar audio
        ruta_audio = self.file_paths[idx]
        y, sr = librosa.load(ruta_audio, sr=None)
        
        # 2. Generar Mel-Espectrograma
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 3. Normalizar la imagen entre 0 y 1 (Vital para que el Autoencoder funcione bien)
        # Sabiendo que los dB suelen ir de -80 a 0.
        mel_spec_normalized = (mel_spec_db + 80) / 80
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
        
        # 4. Ajustar al tamaño fijo (Padding o Truncating)
        # Si el audio es un poco más corto, rellenamos con ceros. Si es más largo, lo cortamos.
        if mel_spec_normalized.shape[1] < self.max_frames:
            pad_width = self.max_frames - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_normalized = mel_spec_normalized[:, :self.max_frames]
            
        # 5. Convertir a Tensor de PyTorch y añadir la dimensión del "Canal" (Color/Grises)
        # Pytorch espera (Canales, Alto, Ancho) -> (1, 128, 320)
        tensor_espectrograma = torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0)
        
        return tensor_espectrograma

# --- PRUEBA DEL DATALOADER ---
if __name__ == "__main__":
    # Recuerda usar la 'r' para la ruta en Windows
    # Apunta a tu carpeta train real dentro de la bomba
    ruta_entrenamiento = r"J:\Master IA\IA Generativa\IAG\dev_data_pump\pump\train"
    
    # Instanciamos el dataset
    dataset = DCASEAudioDataset(data_dir=ruta_entrenamiento)
    
    # Creamos el DataLoader: empaqueta las imágenes de 16 en 16 y las desordena para mejorar el aprendizaje
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Extraemos solo el primer lote para comprobar que todo cuadra
    lote_imagenes = next(iter(dataloader))
    print(f"\nFormato final del Lote: {lote_imagenes.shape}")
    print("Significado: (Batch_Size, Canales, Altura_Mels, Anchura_Tiempo)")