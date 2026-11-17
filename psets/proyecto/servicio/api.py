"""Servicio de riesgo de falla SSD MC1 sobre features ya calculadas en la OBT."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from modelado.pipeline_proyecto import preparar_observacion_inferencia


RUTA_MODELO = Path(__file__).resolve().parent / "modelo.joblib"
app = FastAPI(title="Antes de que falle: SSD MC1")


class ObservacionSSD(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ssd_key: str = Field(min_length=1)
    observation_date: date
    # Ventanas SMART calculadas por la OBT a partir de la telemetría histórica.
    caracteristicas: dict[str, float | None] = Field(min_length=1)


@lru_cache(maxsize=1)
def cargar_modelo() -> dict:
    return joblib.load(RUTA_MODELO)


@app.get("/salud")
def salud() -> dict:
    try:
        paquete = cargar_modelo()
    except (FileNotFoundError, OSError):
        return {"estado": "sin_modelo", "modelo_cargado": False}
    return {"estado": "ok", "modelo_cargado": True, "run_id": paquete["run_id"]}


@app.post("/predecir")
def predecir(observacion: ObservacionSSD, paquete: dict = Depends(cargar_modelo)) -> dict:
    try:
        x = preparar_observacion_inferencia(observacion.caracteristicas, paquete["columnas"])
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    probabilidad = float(paquete["modelo"].predict_proba(x)[0, 1])
    return {
        "ssd_key": observacion.ssd_key,
        "observation_date": observacion.observation_date.isoformat(),
        "probabilidad_falla_30d": probabilidad,
        "requiere_revision": probabilidad >= paquete["umbral"],
        "run_id": paquete["run_id"],
    }
