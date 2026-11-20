"""Servicio de prediccion de la tarifa, instrumentado.

Ademas de responder la tarifa, mide cuanto tarda cada prediccion y con
que codigo responde (las metricas del sistema, en /metricas), y guarda
cada prediccion en predicciones.jsonl, para medir despues cuanto se
equivoco el modelo cuando llegue la tarifa cobrada (desempeno.py).

    uvicorn servicio:app --port 8000
"""

import json
import os
import time
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

from features_taxis import agregar_features
from preparacion_taxis import CATEGORICAS, NUMERICAS, YA_LISTAS

RUTA_MODELO = "modelo_tarifa.joblib"
RUTA_VERSION = "modelo_version.txt"
RUTA_REGISTRO = os.environ.get("RUTA_REGISTRO", "predicciones.jsonl")

modelo = None
version_modelo = None
if os.path.exists(RUTA_MODELO):
    modelo = joblib.load(RUTA_MODELO)
    with open(RUTA_VERSION) as archivo:
        version_modelo = archivo.read().strip()


class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /predecir."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []
        self.error_validacion = 0
        self.error_servidor = 0

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        # [DONE] 2: devuelve un diccionario con "peticiones" (cuantas hubo),
        # "errores_validacion" (cuantas respondieron 422), "errores_servidor"
        # (cuantas respondieron 500 o mas), y "latencia_p50_ms" y
        # "latencia_p95_ms" (np.percentile, redondeadas a 2 decimales).
        # Sin peticiones, las dos latencias son None.

        p95, p99 = np.NaN, np.NaN

        if self.latencias_ms:
            p95, p99 = (round(np.percentile(self.latencias_ms, 95),2),
                        round(np.percentile(self.latencias_ms, 99), 2))

        for codigo in self.codigos:
            if codigo == 422:
                self.error_validacion += 1
            elif codigo >= 500 and codigo < 600:
                self.error_servidor +=1

        return {
            "volumen_peticiones": len(self.latencias_ms),
            "errores_validacion_datos": self.error_validacion,
            "errores_servidor": self.error_servidor,
            "latencia_p95_ms": p95,
            "latencia_p99_ms": p99
        }


metricas = Metricas()
app = FastAPI(title="Tarifa estimada de taxis")


@app.middleware("http")
async def medir(request: Request, call_next):
    """Corre alrededor de cada peticion: lo que esta antes de call_next
    pasa antes de la ruta, y lo que esta despues, cuando ya hay respuesta."""
    if request.url.path != "/predecir":
        return await call_next(request)
    # [DONE] 1: toma el tiempo antes y despues de call_next con time.perf_counter(),
    # y registra en metricas la latencia en milisegundos y el codigo de la
    # respuesta (respuesta.status_code).

    #? Count the time before
    time_before_call_next = time.perf_counter()
    operation_result =  await call_next(request)

    #? Tiempo final
    time_after_call_next = time.perf_counter()
    tiempo_respuesta = (time_after_call_next - time_before_call_next) * 1000
    metricas.registrar(tiempo_respuesta, operation_result.status_code)
    return operation_result


class Viaje(BaseModel):
    """Lo que la app sabe de un viaje antes de que empiece."""
    fecha_hora_inicio: datetime
    zona_origen: str
    zona_destino: str
    distancia_km: float | None = None
    pasajeros: float | None = None
    metodo_pago: str | None = None


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.post("/predecir")
def predecir(viaje: Viaje):
    datos = agregar_features(pd.DataFrame([viaje.model_dump()]))
    tarifa = round(float(modelo.predict(datos[NUMERICAS + YA_LISTAS + CATEGORICAS])[0]), 2)
    id_prediccion = uuid.uuid4().hex
    # [Done]  3: agrega una linea a RUTA_REGISTRO (abre el archivo con "a") con
    # json.dumps de un diccionario con id_prediccion, "momento"
    # (datetime.now().isoformat()), version_modelo, los campos del viaje
    # (viaje.model_dump(mode="json")) y "tarifa_estimada".

    with open(RUTA_REGISTRO,'a') as registro_predicciones:
        registro_on_request = {
            "id_prediccion": id_prediccion,
            "tarifa": tarifa,
            "prediccion_created_at": datetime.now().isoformat(),
            "version_modelo": version_modelo,
        }

        for key, value in viaje.__dict__.items():
            registro_on_request[key] = str(value)

        registro_predicciones.write(json.dumps(registro_on_request) + "\n")
    return {"id_prediccion": id_prediccion, "tarifa_estimada": tarifa, "version_modelo": version_modelo}
