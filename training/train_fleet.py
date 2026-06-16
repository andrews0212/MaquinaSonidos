import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import DCASEAudioDataset
from core.model import AutoencoderCNN

def calcular_umbral_3sigma(modelo, dataloader, device):
    modelo.eval()
    criterio = nn.MSELoss(reduction='none')
    errores = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            outputs = modelo(inputs)
            loss = criterio(outputs, inputs)
            mse_por_muestra = loss.mean(dim=[1, 2, 3]).cpu().numpy()
            errores.extend(mse_por_muestra)
            
    errores = np.array(errores)
    mean_loss = np.mean(errores)
    std_loss = np.max([np.std(errores), 1e-6])
    umbral = mean_loss + (3 * std_loss)
    return float(umbral)

def entrenar_toda_la_flota(ruta_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carpetas = [c for c in os.listdir(ruta_base) if c.startswith("dev_data_")]
    configuracion_flota = {}

    for carpeta in carpetas:
        tipo_maquina = carpeta.split("_")[-1]
        print(f"\n========================================")
        print(f"[{tipo_maquina.upper()}] - Iniciando entrenamiento experto")
        
        # Lógica de resolución de rutas DCASE anidadas
        ruta_train = os.path.join(ruta_base, carpeta, "train")
        if not os.path.exists(ruta_train):
            ruta_train = os.path.join(ruta_base, carpeta, tipo_maquina, "train")
            
        if not os.path.exists(ruta_train):
            print(f"⚠️ No se encontró la carpeta 'train' para {tipo_maquina}. Saltando...")
            continue

        dataset = DCASEAudioDataset(data_dir=ruta_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        
        modelo = AutoencoderCNN().to(device)
        optimizer = optim.Adam(modelo.parameters(), lr=1e-3)
        criterio_train = nn.MSELoss()
        
        for epoch in range(15):
            modelo.train()
            for batch in dataloader:
                inputs = batch.to(device)
                optimizer.zero_grad()
                outputs = modelo(inputs)
                loss = criterio_train(outputs, inputs)
                loss.backward()
                optimizer.step()
        
        umbral_optimo = calcular_umbral_3sigma(modelo, dataloader, device)
        path_modelo = os.path.join("models", f"modelo_{tipo_maquina}_experto.pth")
        os.makedirs("models", exist_ok=True)
        torch.save(modelo.state_dict(), path_modelo)
        
        configuracion_flota[tipo_maquina] = {
            "modelo_path": path_modelo,
            "umbral_alarma": umbral_optimo
        }
        print(f"-> Guardado {path_modelo} | Umbral 3-Sigma: {umbral_optimo:.6f}")
        
    with open("fleet_config.json", "w") as f:
        json.dump(configuracion_flota, f, indent=4)

if __name__ == "__main__":
    entrenar_toda_la_flota(r"J:\Master IA\IA Generativa\IAG")