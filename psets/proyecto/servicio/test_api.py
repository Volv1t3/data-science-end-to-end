"""Pruebas del contrato HTTP sin modelo de producción ni archivo de datos."""

import json

from fastapi.testclient import TestClient
import numpy as np

from modelado.pipeline_proyecto import NUMERICAS
from servicio import api
from servicio.api import app, cargar_modelo


class ModeloPrueba:
    def predict_proba(self, x):
        assert x.columns.tolist() == [NUMERICAS[1], NUMERICAS[0]]
        assert x.iloc[0].tolist() == [2.0, 1.0]
        return np.array([[0.2, 0.8]])


def paquete_prueba():
    return {"modelo": ModeloPrueba(), "columnas": [NUMERICAS[1], NUMERICAS[0]],
            "umbral": 0.5, "run_id": "corrida-de-prueba", "version_modelo": "99"}


def test_salud_informa_la_corrida(monkeypatch):
    monkeypatch.setattr(api, "cargar_modelo", paquete_prueba)
    with TestClient(app) as cliente:
        respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True,
                              "run_id": "corrida-de-prueba", "version_modelo": "99"}


def test_prediccion_valida_usa_orden_y_queda_registrada(tmp_path, monkeypatch):
    ruta = tmp_path / "predicciones.jsonl"
    monkeypatch.setattr(api, "RUTA_REGISTRO", ruta)
    app.dependency_overrides[cargar_modelo] = paquete_prueba
    try:
        with TestClient(app) as cliente:
            respuesta = cliente.post("/predecir", json={
                "ssd_key": "SSD-1", "observation_date": "2019-11-15",
                "caracteristicas": {NUMERICAS[0]: 1, NUMERICAS[1]: 2},
            })
        assert respuesta.status_code == 200
        assert respuesta.json()["probabilidad_falla_30d"] == 0.8
        assert respuesta.json()["requiere_revision"] is True
        assert respuesta.json()["run_id"] == "corrida-de-prueba"
        assert respuesta.json()["version_modelo"] == "99"
        registro = json.loads(ruta.read_text(encoding="utf-8").strip())
        assert registro["id_prediccion"] == respuesta.json()["id_prediccion"]
        assert registro["entrada"]["ssd_key"] == "SSD-1"
        assert registro["entrada"]["observation_date"] == "2019-11-15"
        assert registro["salida"]["probabilidad_falla_30d"] == 0.8
        assert registro["version_modelo"] == "99"
    finally:
        app.dependency_overrides.clear()


def test_entrada_invalida_devuelve_422():
    app.dependency_overrides[cargar_modelo] = paquete_prueba
    try:
        with TestClient(app) as cliente:
            respuesta = cliente.post("/predecir", json={
                "ssd_key": "SSD-1", "observation_date": "2019-11-15",
                "caracteristicas": {NUMERICAS[0]: 1},
            })
        assert respuesta.status_code == 422
        assert "Faltan características" in respuesta.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_metricas_separan_validacion_y_error_del_servidor(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "metricas", api.MetricasServicio())
    monkeypatch.setattr(api, "RUTA_REGISTRO", tmp_path / "predicciones.jsonl")
    peticion = {
        "ssd_key": "SSD-1", "observation_date": "2019-11-15",
        "caracteristicas": {NUMERICAS[0]: 1, NUMERICAS[1]: 2},
    }

    class ModeloFallido:
        def predict_proba(self, x):
            raise RuntimeError("fallo del estimador")

    app.dependency_overrides[cargar_modelo] = paquete_prueba
    try:
        with TestClient(app, raise_server_exceptions=False) as cliente:
            assert cliente.get("/metricas").json()["volumen_peticiones"] == 0
            assert cliente.post("/predecir", json={**peticion, "caracteristicas": {}}).status_code == 422
            app.dependency_overrides[cargar_modelo] = lambda: {
                **paquete_prueba(), "modelo": ModeloFallido()
            }
            assert cliente.post("/predecir", json=peticion).status_code == 500
            app.dependency_overrides[cargar_modelo] = paquete_prueba
            assert cliente.post("/predecir", json=peticion).status_code == 200
            resumen = cliente.get("/metricas").json()
        assert resumen["volumen_peticiones"] == 3
        assert resumen["errores_validacion_datos"] == 1
        assert resumen["errores_servidor"] == 1
        assert 0 <= resumen["latencia_p50_ms"] <= resumen["latencia_p95_ms"]
    finally:
        app.dependency_overrides.clear()
