from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="Sonitel Central Ingestion Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AlertaPayload(BaseModel):
    id_planta: str
    id_maquina: str
    tipo_maquina: str
    error_mse: float
    umbral_limite: float
    espectrograma_data: List[List[float]]

class NuevaMaquina(BaseModel):
    id_planta: str
    id_maquina: str
    tipo_maquina: str

ALERTAS_ACTIVAS = {}
INVENTARIO_FLOTA = {}

@app.post("/api/alerts")
def recibir_alerta_edge(alerta: AlertaPayload):
    id_unico = f"{alerta.id_planta}_{alerta.id_maquina}"
    ALERTAS_ACTIVAS[id_unico] = {**alerta.model_dump(), "estado_verificacion": "PENDIENTE_OPERARIO"}
    return {"status": "procesado"}

@app.get("/api/dashboard/active")
def obtener_alertas_dashboard():
    return list(ALERTAS_ACTIVAS.values())

@app.post("/api/dashboard/verify/{id_registro}")
def verificar_alerta(id_registro: str, accion: str):
    if id_registro not in ALERTAS_ACTIVAS:
        raise HTTPException(status_code=404)
        
    if accion == "CONFIRMAR":
        ALERTAS_ACTIVAS[id_registro]["estado_verificacion"] = "TECNICO_DESPLEGADO"
    elif accion == "DESCARTAR":
        ALERTAS_ACTIVAS.pop(id_registro)
        
    return {"status": "actualizado"}

@app.post("/api/inventory/register")
def registrar_maquina(maquina: NuevaMaquina):
    id_unico = f"{maquina.id_planta}_{maquina.id_maquina}"
    INVENTARIO_FLOTA[id_unico] = {
        "id_planta": maquina.id_planta,
        "id_maquina": maquina.id_maquina,
        "tipo_maquina": maquina.tipo_maquina,
        "estado": "Esperando telemetría Edge"
    }
    return {"status": "registrado", "id_registro": id_unico, "detalles": INVENTARIO_FLOTA[id_unico]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)