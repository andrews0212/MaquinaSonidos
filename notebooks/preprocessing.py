import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualizar_espectrograma(ruta_audio, titulo="Mel-Espectrograma"):
    """
    Toma un archivo .wav y dibuja su Mel-Espectrograma.
    """
    # 1. Cargar el audio 
    # sr=None le dice a librosa que mantenga la frecuencia de muestreo original del .wav
    y, sr = librosa.load(ruta_audio, sr=None)
    
    # 2. Generar el Mel-Espectrograma
    # n_fft: tamaño de la ventana (resolución en frecuencia)
    # hop_length: salto entre ventanas (resolución en tiempo)
    # n_mels: número de bandas de frecuencia (altura de nuestra imagen final)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    
    # 3. Convertir la amplitud a Decibelios (escala logarítmica, imita al oído humano)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 4. Dibujar la imagen
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

# --- PRUEBA DEL CÓDIGO ---
# Navega por tu carpeta 'dev_data_pump'. 
# Busca un audio dentro de la carpeta 'train' (que son todos normales).
# Sustituye la ruta de abajo por la ruta real de tu ordenador.

ruta_prueba_normal = r"J:\Master IA\IA Generativa\IAG\dev_data_pump\pump\test\anomaly_id_00_00000006.wav"
visualizar_espectrograma(ruta_prueba_normal, "Bomba - Funcionamiento Normal")