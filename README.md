# MaquinaSonido — Predictive Maintenance Edge AI

> Sistema de mantenimiento predictivo basado en **detección de anomalías acústicas** para maquinaria industrial. Desplegable en dispositivos Edge (IoT), sin dependencia de conectividad constante.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat&logo=fastapi&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-DCASE%202020-blueviolet?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Descripción

La mayoría de los fallos en maquinaria industrial se manifiestan antes de ser visibles: como una variación en el sonido. Este proyecto implementa un pipeline completo de **Acoustic Anomaly Detection** usando autoencoders convolucionales entrenados exclusivamente con sonido normal. Cuando el modelo no puede reconstruir fielmente un audio de entrada, lo clasifica como anomalía.

El sistema corre directamente en el borde de la red (Edge), enviando alertas solo cuando detecta un fallo. El Centro de Operaciones de Red (NOC) recibe las alertas con el espectrograma visual del evento para que un operario tome la decisión final (Human-in-the-Loop).

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        PLANTA INDUSTRIAL                        │
│                                                                 │
│  [Micrófono]                                                    │
│      │                                                          │
│      ▼                                                          │
│  [edge_daemon.py]  ←── fleet_config.json (umbral + modelo)     │
│      │                                                          │
│      │  Audio → Mel Spectrogram → CNN Autoencoder              │
│      │  MSE > Umbral 3-Sigma?                                   │
│      │                                                          │
│      │  SÍ: POST /api/alerts ──────────────────────────────┐   │
│      │  NO: log local "OK"                                 │   │
└─────────────────────────────────────────────────────────────────┘
                                                             │
                    ┌────────────────────────────────────────┘
                    ▼
         ┌──────────────────────┐
         │  backend_api.py      │  FastAPI — Puerto 8001
         │  (Central NOC)       │
         └──────────┬───────────┘
                    │  GET /api/dashboard/active (polling 3s)
                    ▼
         ┌──────────────────────┐
         │  dashboard.html      │  NOC — Human in the Loop
         │                      │
         │  [Espectrograma]     │  ┌─────────────────────┐
         │  [Métricas MSE]      │  │ ✖ Falso Positivo    │
         │                      │  │ ⚙ Desplegar Técnico │
         └──────────────────────┘  └─────────────────────┘
```

---

## Características Principales

- **Detección No Supervisada** — El autoencoder aprende exclusivamente con audio normal. No necesita ejemplos de fallos etiquetados.
- **Umbral Estadístico (3-Sigma)** — El umbral de alarma se calcula automáticamente por máquina usando la distribución de errores de reconstrucción.
- **Flota de Expertos** — Un modelo especializado por tipo de máquina (`pump`, `valve`, `fan`…), cada uno con su propio umbral calibrado.
- **Edge-First** — El demonio IoT no requiere GPU. Toda la inferencia ocurre en CPU local; el backend solo recibe alertas positivas.
- **Validación HITL** — El dashboard NOC muestra el mel-espectrograma del evento para que el operario confirme o descarte la alerta antes de desplegar un técnico.

---

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| **Modelo ML** | PyTorch — CNN Autoencoder (Encoder/Decoder simétrico) |
| **Procesado de Audio** | Librosa — Mel Spectrogram (128 bandas, 320 frames) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend NOC** | HTML5 / Canvas API / Vanilla JS |
| **Dataset** | DCASE 2020 Task 2 — Unsupervised Detection of Anomalous Sounds |

---

## Estructura del Proyecto

```
MaquinaSonido/
├── core/               # Módulos ML reutilizables
│   ├── model.py        #   Arquitectura CNN Autoencoder
│   └── dataset.py      #   PyTorch Dataset (audio → mel spectrogram)
│
├── training/           # Scripts de entrenamiento
│   ├── train.py        #   Entrenamiento de una sola máquina
│   └── train_fleet.py  #   Entrenamiento de toda la flota + calibración 3-sigma
│
├── edge/               # Demonio IoT (producción)
│   └── edge_daemon.py  #   Inferencia + envío de alertas al backend
│
├── api/                # Backend central (NOC)
│   └── backend_api.py  #   FastAPI: ingestión de alertas e inventario de flota
│
├── frontend/           # Dashboard de operaciones
│   └── dashboard.html  #   Panel de excepciones con espectrogramas y acciones HITL
│
├── tests/              # Scripts de prueba y demo
│   ├── inference.py    #   Demo de inferencia sobre un audio individual
│   └── test_fleet_anomalies.py  # Simulación de estrés de la flota completa
│
├── notebooks/          # Exploración y presentación
│   ├── preprocessing.py
│   └── Presentacion_Sonitel_Industrial.ipynb
│
├── data/               # Datos de audio (DCASE)
│   ├── fan/id_XX/{normal,abnormal}/
│   ├── pump/id_XX/{normal,abnormal}/
│   └── valve/id_XX/{normal,abnormal}/
│
├── models/             # Modelos entrenados (.pth) — generados por train_fleet.py
├── fleet_config.json   # Configuración de flota: rutas de modelos y umbrales
└── requirements.txt
```

---

## Cómo Funciona el Modelo

```
Audio .wav
    │
    ▼  librosa.feature.melspectrogram()
Mel Spectrogram  [128 × 320]  (normalizado a [0, 1])
    │
    ▼  CNN Encoder (Conv2d × 3, stride 2)
Representación latente comprimida
    │
    ▼  CNN Decoder (ConvTranspose2d × 3)
Reconstrucción del espectrograma
    │
    ▼  MSE(original, reconstrucción)
Error de Reconstrucción
    │
    ├─ ≤ Umbral  →  ✅ Máquina Normal
    └─ > Umbral  →  🔴 Anomalía Detectada
```

El modelo nunca ve audio anómalo durante el entrenamiento. Al encontrar un patrón sonoro desconocido, su error de reconstrucción excede el umbral estadístico (media + 3σ calculado sobre el set de validación normal).

---

## Instalación

```bash
git clone https://github.com/tu-usuario/MaquinaSonido.git
cd MaquinaSonido

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU (opcional):** Si tienes NVIDIA CUDA, instala PyTorch manualmente desde [pytorch.org](https://pytorch.org/get-started/locally/) antes de ejecutar `pip install -r requirements.txt`.

---

## Uso

### 1. Entrenamiento de la flota completa

Coloca los datos de audio (formato DCASE) en `data/` y ejecuta desde la raíz del proyecto:

```bash
python training/train_fleet.py
```

Genera un modelo `.pth` por tipo de máquina en `models/` y actualiza `fleet_config.json` con los umbrales calibrados.

### 2. Levantar el backend NOC

```bash
python api/backend_api.py
# API disponible en http://127.0.0.1:8001
```

### 3. Abrir el dashboard

Abre `frontend/dashboard.html` en el navegador. Se actualiza automáticamente cada 3 segundos consultando el backend.

### 4. Simular una alerta desde el Edge

```bash
python edge/edge_daemon.py \
  --planta "Planta_Sevilla" \
  --maquina "BOMBA-01" \
  --tipo "pump" \
  --audio "data/pump/id_00/abnormal/archivo.wav"
```

---

## Dataset

Este proyecto usa el dataset público **DCASE 2020 Challenge Task 2**:
> *Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring*

[Descargar dataset oficial →](https://dcase.community/challenge2020/task2-unsupervised-detection-of-anomalous-sounds)

Tipos de máquina soportados: `fan`, `pump`, `valve`, `slider`, `ToyCar`, `ToyConveyor`.

---

## Autor

**Andrews** — [andrewsdosramos@gmail.com](mailto:andrewsdosramos@gmail.com)
