import os
import subprocess
import time

def ejecutar_test_flota():
    base_dir = r"J:\Master IA\IA Generativa\IAG"
    maquinas = ['pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    
    print("--- Iniciando Simulación de Estrés Edge (IoT) ---")
    
    for maquina in maquinas:
        # Búsqueda del archivo anómalo
        test_dir = os.path.join(base_dir, f"dev_data_{maquina}", maquina, "test")
        
        if not os.path.exists(test_dir):
            print(f"Directorio de test no encontrado para: {maquina}")
            continue
            
        archivos = os.listdir(test_dir)
        archivo_anomalo = next((f for f in archivos if f.startswith("anomaly")), None)
        
        if not archivo_anomalo:
            print(f"No se encontraron archivos de anomalía para {maquina}")
            continue
            
        ruta_audio = os.path.join(test_dir, archivo_anomalo)
        id_maquina_test = f"Test_Falla_{maquina.upper()}"
        
        print(f"\n[+] Disparando Demonio Edge para: {maquina}")
        
        # Ejecución del subproceso dinámico
        comando = [
            "python", "edge_daemon.py",
            "--planta", "Planta_Pruebas_Global",
            "--maquina", id_maquina_test,
            "--tipo", maquina,
            "--audio", ruta_audio
        ]
        
        try:
            subprocess.run(comando, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar el demonio para {maquina}: {e}")
            
        time.sleep(3) # Pausa para renderizado en el Dashboard

if __name__ == "__main__":
    ejecutar_test_flota()
    print("\nSimulación finalizada. Comprueba el Dashboard NOC.")