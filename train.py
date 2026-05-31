import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importamos nuestros propios módulos
from dataset import DCASEAudioDataset
from model import AutoencoderCNN

def entrenar_modelo():
    # 1. Configuración del Entorno (Detectar GPU si existe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando entrenamiento en: {device} ---")

    # 2. Hiperparámetros
    EPOCHS = 20         # Veces que la red verá todo el dataset completo
    BATCH_SIZE = 16      # Paquetes de imágenes por iteración
    LEARNING_RATE = 1e-3 # Tamaño del paso del optimizador
    
    # IMPORTANTE: Cambia esta ruta a tu carpeta real
    RUTA_TRAIN = r"J:\Master IA\IA Generativa\IAG\dev_data_pump\pump\train"

    # 3. Inicializar Datos, Modelo, Pérdida y Optimizador
    dataset = DCASEAudioDataset(data_dir=RUTA_TRAIN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    modelo = AutoencoderCNN().to(device)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    mejor_loss = float('inf')

    # 4. Bucle de Entrenamiento
    for epoca in range(EPOCHS):
        modelo.train()
        loss_acumulado = 0.0
        
        for batch in dataloader:
            # Mandar datos a la CPU/GPU
            imagenes_reales = batch.to(device)
            
            # Paso 1: Forward (La red intenta dibujar la imagen)
            imagenes_reconstruidas = modelo(imagenes_reales)
            
            # Paso 2: Calcular el error
            loss = criterio(imagenes_reconstruidas, imagenes_reales)
            
            # Paso 3: Backpropagation y optimización
            optimizador.zero_grad() # Limpiar memoria de gradientes
            loss.backward()         # Calcular derivadas
            optimizador.step()      # Actualizar pesos
            
            loss_acumulado += loss.item()
            
        # Calcular el error medio de esta época
        loss_promedio = loss_acumulado / len(dataloader)
        print(f"Época [{epoca+1}/{EPOCHS}] | MSE Loss: {loss_promedio:.6f}")
        
        # 5. Guardar el modelo si es el mejor hasta ahora
        if loss_promedio < mejor_loss:
            mejor_loss = loss_promedio
            torch.save(modelo.state_dict(), "autoencoder_bomba_mejor.pth")
            print("  -> ¡Nuevo mejor modelo guardado!")

    print("--- Entrenamiento Finalizado ---")

if __name__ == "__main__":
    entrenar_modelo()