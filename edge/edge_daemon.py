import argparse
import json
import requests
import torch
import numpy as np
import librosa
from core.model import AutoencoderCNN

URL_BACKEND = "http://127.0.0.1:8001/api/alerts"

def cargar_sistema_edge(tipo_maquina):
    try:
        with open("fleet_config.json", "r") as f:
            config = json.load(f)[tipo_maquina]
        
        modelo = AutoencoderCNN()
        modelo.load_state_dict(torch.load(config["modelo_path"], map_location="cpu", weights_only=True))
        modelo.eval()
        return modelo, config["umbral_alarma"]
    except KeyError:
        print(f"[ERROR CRÍTICO] La máquina '{tipo_maquina}' no existe en fleet_config.json.")
        exit(1)

def procesar_buffer_audio(ruta_archivo, modelo, umbral, id_planta, id_maquina, tipo_maquina):
    y, sr = librosa.load(ruta_archivo, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_normalized = np.clip((librosa.power_to_db(mel_spec, ref=np.max) + 80) / 80, 0, 1)
    
    if mel_spec_normalized.shape[1] < 320:
        pad_width = 320 - mel_spec_normalized.shape[1]
        mel_spec_normalized = np.pad(mel_spec_normalized, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_normalized = mel_spec_normalized[:, :320]
        
    tensor_input = torch.tensor(mel_spec_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        mse_error = torch.mean((modelo(tensor_input) - tensor_input) ** 2).item()
        
    if mse_error > umbral:
        payload = {
            "id_planta": id_planta, 
            "id_maquina": id_maquina, 
            "tipo_maquina": tipo_maquina,
            "error_mse": mse_error, 
            "umbral_limite": umbral, 
            "espectrograma_data": mel_spec_normalized.tolist()
        }
        try:
            requests.post(URL_BACKEND, json=payload, timeout=5)
            print(f"🔴 [ALERTA ENVIADA] Anomalía en {id_maquina} | MSE: {mse_error:.6f} > Umbral: {umbral:.6f}")
        except Exception as e:
            print(f"⚠️ [ERROR RED] Backend inalcanzable. Detalles: {e}")
    else:
        print(f"✅ [OK] {id_maquina} operando con normalidad. MSE: {mse_error:.6f} (Umbral: {umbral:.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonio IoT Sonitel Industrial")
    parser.add_argument("--planta", type=str, required=True, help="ID de la planta")
    parser.add_argument("--maquina", type=str, required=True, help="ID físico de la máquina")
    parser.add_argument("--tipo", type=str, required=True, help="Tipo de máquina en el dataset")
    parser.add_argument("--audio", type=str, required=True, help="Ruta al archivo .wav capturado")
    args = parser.parse_args()

    modelo_local, umbral_local = cargar_sistema_edge(args.tipo)
    print(f"Iniciando demonio para {args.maquina} ({args.tipo}) en {args.planta}...")
    procesar_buffer_audio(args.audio, modelo_local, umbral_local, args.planta, args.maquina, args.tipo)