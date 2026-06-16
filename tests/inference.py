import torch
import torch.nn as nn
import librosa
import numpy as np

# Importamos la arquitectura vacía para luego rellenarla con los pesos
from core.model import AutoencoderCNN

def preprocesar_audio_inferencia(ruta_audio, max_frames=320):
    """Hace exactamente lo mismo que el DataLoader, pero para un solo archivo al vuelo."""
    y, sr = librosa.load(ruta_audio, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    mel_spec_normalized = (mel_spec_db + 80) / 80
    mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
    
    if mel_spec_normalized.shape[1] < max_frames:
        pad_width = max_frames - mel_spec_normalized.shape[1]
        mel_spec_normalized = np.pad(mel_spec_normalized, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_normalized = mel_spec_normalized[:, :max_frames]
        
    # Añadimos la dimensión del Batch (1) y el Canal (1) -> (1, 1, 128, 320)
    tensor = torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def evaluar_maquina(ruta_audio, ruta_modelo="models/autoencoder_bomba_mejor.pth"):
    # 1. Cargar el modelo en CPU (En Edge normalmente no hay GPU potentes)
    device = torch.device("cpu")
    modelo = AutoencoderCNN()
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=device, weights_only=True))
    modelo.eval() # Modo evaluación (apaga Dropout, BatchNormalization, etc.)
    
    # 2. Preprocesar el audio que acaba de entrar
    tensor_entrada = preprocesar_audio_inferencia(ruta_audio)
    
    # 3. Inferencia (Paso por la red)
    criterio = nn.MSELoss()
    with torch.no_grad(): # No calculamos gradientes, ahorra muchísima RAM
        reconstruccion = modelo(tensor_entrada)
        error_mse = criterio(reconstruccion, tensor_entrada).item()
        
    return error_mse

if __name__ == "__main__":
    # TODO: Tienes que buscar en tu carpeta 'test' (NO en la de train)
    # 1. Un archivo que se llame normal_id...
    ruta_normal = r"J:\Master IA\IA Generativa\IAG\dev_data_pump\pump\test\normal_id_00_00000000.wav"
    
    # 2. Un archivo que se llame anomaly_id...
    ruta_anomalia = r"J:\Master IA\IA Generativa\IAG\dev_data_pump\pump\test\anomaly_id_06_00000101.wav"
    
    print("--- Evaluación de Inferencia (Edge) ---")
    mse_normal = evaluar_maquina(ruta_normal)
    print(f"Error MSE en audio NORMAL  : {mse_normal:.6f}")
    
    mse_anomalo = evaluar_maquina(ruta_anomalia)
    print(f"Error MSE en audio ANÓMALO : {mse_anomalo:.6f}")