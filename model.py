import torch
import torch.nn as nn

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super(AutoencoderCNN, self).__init__()

        # 1. ENCODER: El "embudo" que comprime la imagen
        # Entra 1 canal (imagen en escala de grises), salen 16, luego 32, luego 64.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # 2. DECODER: El "pintor" que intenta reconstruir la imagen original
        # Hace el proceso inverso: de 64 canales baja a 32, luego a 16, y vuelve a 1 canal.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Usamos Sigmoid asumiendo que normalizaremos los píxeles entre 0 y 1
        )

    def forward(self, x):
        # El flujo de los datos a través de la red
        latente = self.encoder(x)
        reconstruccion = self.decoder(latente)
        return reconstruccion

# --- PRUEBA RÁPIDA DE LA ARQUITECTURA ---
if __name__ == "__main__":
    # Simulamos un tensor que representa un "lote" de 8 espectrogramas
    # Formato PyTorch: (Batch_Size, Channels, Height, Width) -> (8, 1, 128, 313)
    espectrograma_simulado = torch.randn(8, 1, 128, 313)

    modelo = AutoencoderCNN()
    salida = modelo(espectrograma_simulado)

    print(f"Formato de entrada: {espectrograma_simulado.shape}")
    print(f"Formato de salida : {salida.shape}")
    # Si ambas formas coinciden, nuestra red está bien estructurada.
