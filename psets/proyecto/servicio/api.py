"""Servicio de riesgo de falla SSD MC1 sobre features ya calculadas en la OBT."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from modelado.pipeline_proyecto import preparar_observacion_inferencia


RUTA_MODELO = Path(__file__).resolve().parent / "modelo.joblib"
RUTA_VERSION = RUTA_MODELO.with_name("modelo_version.txt")
RUTA_REGISTRO = Path(
    os.getenv(
        "MC1_PREDICCIONES_PATH",
        str(Path(__file__).resolve().parents[1] / "monitoreo/predicciones.jsonl"),
    )
)
_LOCK_REGISTRO = Lock()
app = FastAPI(title="Antes de que falle: SSD MC1")


class MetricasServicio:
    """Contadores acumulados y percentiles de las últimas 10 000 predicciones."""

    def __init__(self) -> None:
        self.peticiones = 0
        self.errores_validacion = 0
        self.errores_servidor = 0
        self.latencias_ms: deque[float] = deque(maxlen=10_000)
        self._lock = Lock()

    def registrar(self, latencia_ms: float, codigo: int) -> None:
        with self._lock:
            self.peticiones += 1
            self.errores_validacion += codigo == 422
            self.errores_servidor += codigo >= 500
            self.latencias_ms.append(latencia_ms)

    def resumen(self) -> dict:
        with self._lock:
            latencias = list(self.latencias_ms)
            return {
                "volumen_peticiones": self.peticiones,
                "errores_validacion_datos": self.errores_validacion,
                "errores_servidor": self.errores_servidor,
                "latencia_p50_ms": round(float(np.percentile(latencias, 50)), 2) if latencias else None,
                "latencia_p95_ms": round(float(np.percentile(latencias, 95)), 2) if latencias else None,
            }


metricas = MetricasServicio()


@app.middleware("http")
async def medir_prediccion(request: Request, call_next):
    if request.url.path != "/predecir":
        return await call_next(request)
    inicio = time.perf_counter()
    codigo = 500
    try:
        respuesta = await call_next(request)
        codigo = respuesta.status_code
        return respuesta
    finally:
        metricas.registrar((time.perf_counter() - inicio) * 1000, codigo)


class ObservacionSSD(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ssd_key: str = Field(min_length=1)
    observation_date: date
    # Ventanas SMART calculadas por la OBT a partir de la telemetría histórica.
    caracteristicas: dict[str, float | None] = Field(min_length=1)


@lru_cache(maxsize=1)
def cargar_modelo() -> dict:
    paquete = joblib.load(RUTA_MODELO)
    version = RUTA_VERSION.read_text(encoding="utf-8").strip()
    if not version or version != paquete.get("version_modelo"):
        raise ValueError("El modelo y su archivo de versión no coinciden")
    return paquete


@app.get("/salud")
def salud() -> dict:
    try:
        paquete = cargar_modelo()
    except (FileNotFoundError, OSError, ValueError):
        return {"estado": "sin_modelo", "modelo_cargado": False}
    return {"estado": "ok", "modelo_cargado": True,
            "run_id": paquete["run_id"], "version_modelo": paquete["version_modelo"]}


@app.get("/metricas")
def ver_metricas() -> dict:
    return metricas.resumen()


@app.post("/predecir")
def predecir(observacion: ObservacionSSD, paquete: dict = Depends(cargar_modelo)) -> dict:
    try:
        x = preparar_observacion_inferencia(observacion.caracteristicas, paquete["columnas"])
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    probabilidad = float(paquete["modelo"].predict_proba(x)[0, 1])
    resultado = {
        "id_prediccion": uuid.uuid4().hex,
        "ssd_key": observacion.ssd_key,
        "observation_date": observacion.observation_date.isoformat(),
        "probabilidad_falla_30d": probabilidad,
        "requiere_revision": probabilidad >= paquete["umbral"],
        "run_id": paquete["run_id"],
        "version_modelo": paquete["version_modelo"],
    }
    registro = {
        "id_prediccion": resultado["id_prediccion"],
        "momento_utc": datetime.now(timezone.utc).isoformat(),
        "version_modelo": paquete["version_modelo"],
        "run_id": paquete["run_id"],
        "entrada": observacion.model_dump(mode="json"),
        "salida": resultado,
    }
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK_REGISTRO, RUTA_REGISTRO.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False, allow_nan=False) + "\n")
    return resultado
