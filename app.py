"""
Monitor de Bomba en Tiempo Real
Inicio: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
import io
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
from model import AutoencoderCNN

# ── Parámetros de audio (deben coincidir con el entrenamiento) ─────────────────
SR          = 16_000
N_FFT       = 2_048
HOP         = 512
N_MELS      = 128
MAX_FRAMES  = 320                               # tamaño exacto de entrenamiento
BUF_SAMPLES = (MAX_FRAMES - 1) * HOP           # 163 328 samples ≈ 10.2 s
INFER_EVERY = HOP * 16                          # cadencia de inferencia: 8 192 samples ≈ 0.51 s

# ── Carga del modelo ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = AutoencoderCNN().to(DEVICE)
_model.load_state_dict(
    torch.load("autoencoder_bomba_mejor.pth", map_location=DEVICE, weights_only=False)
)
_model.eval()
print(f"[OK] Modelo 'autoencoder_bomba_mejor' cargado en {DEVICE}")

# ── Pipeline de inferencia ─────────────────────────────────────────────────────
def infer(samples: np.ndarray) -> float:
    """PCM float32 a 16 kHz  →  MSE de reconstrucción."""
    mel = librosa.feature.melspectrogram(
        y=samples, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    ref  = float(np.max(mel)) if np.max(mel) > 0 else 1e-10
    db   = librosa.power_to_db(mel, ref=ref)
    norm = np.clip((db + 80) / 80, 0.0, 1.0).astype(np.float32)

    if norm.shape[1] < MAX_FRAMES:
        norm = np.pad(norm, ((0, 0), (0, MAX_FRAMES - norm.shape[1])))
    else:
        norm = norm[:, :MAX_FRAMES]

    t = torch.tensor(norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        rec = _model(t)
    return float(F.mse_loss(rec, t))

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Monitor Bomba")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return HTMLResponse(
        (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    )


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()

    buf   = np.empty(0, dtype=np.float32)
    since = 0
    loop  = asyncio.get_event_loop()

    try:
        async for raw in websocket.iter_bytes():
            chunk  = np.frombuffer(raw, dtype=np.float32).copy()
            buf    = np.concatenate([buf, chunk])
            since += len(chunk)

            # Mantener buffer acotado
            if len(buf) > BUF_SAMPLES + INFER_EVERY:
                buf = buf[-BUF_SAMPLES:]

            if since >= INFER_EVERY:
                since  = 0
                window = buf[-BUF_SAMPLES:] if len(buf) >= BUF_SAMPLES else buf.copy()
                mse    = await loop.run_in_executor(None, infer, window)

                await websocket.send_text(json.dumps({
                    "mse":      round(mse, 7),
                    "buffer_s": round(len(window) / SR, 2),
                    "max_s":    round(BUF_SAMPLES  / SR, 2),
                    "ready":    len(buf) >= BUF_SAMPLES // 2,
                }))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[WS ERROR] {exc}")


@app.post("/infer-file")
async def infer_file(audio: UploadFile = File(...)):
    """Analiza un archivo WAV/MP3 completo con ventana deslizante y devuelve los MSE."""
    content = await audio.read()
    try:
        y, _ = librosa.load(io.BytesIO(content), sr=SR, mono=True)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"No se pudo leer el audio: {exc}"})

    if len(y) == 0:
        return JSONResponse(status_code=400, content={"error": "Archivo de audio vacío"})

    loop = asyncio.get_event_loop()

    # Ventana deslizante con 50 % de solapamiento
    step       = BUF_SAMPLES // 2
    mse_values = []

    if len(y) <= BUF_SAMPLES:
        mse_values.append(round(await loop.run_in_executor(None, infer, y), 7))
    else:
        for start in range(0, len(y) - BUF_SAMPLES + 1, step):
            window = y[start : start + BUF_SAMPLES]
            mse_values.append(round(await loop.run_in_executor(None, infer, window), 7))

    mse_arr = np.array(mse_values)
    return {
        "filename":         audio.filename,
        "duration_s":       round(len(y) / SR, 2),
        "windows":          len(mse_values),
        "mse_max":          round(float(mse_arr.max()),  7),
        "mse_mean":         round(float(mse_arr.mean()), 7),
        "mse_std":          round(float(mse_arr.std()),  7),
        "umbral_sugerido":  round(float(mse_arr.mean() + 3 * mse_arr.std()), 7),
        "mse_values":       mse_values,
    }
