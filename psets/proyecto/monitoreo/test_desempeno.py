"""Prueba el retraso de etiquetas y la comparación por versión sin archivos reales."""

import json
from datetime import date

import pandas as pd

from modelado.pipeline_proyecto import CLAVE_SSD, FECHA, OBJETIVO
from monitoreo.desempeno import cargar_predicciones, cargar_verdad, construir_reporte


def test_pr_auc_usa_solo_verdad_madura_y_separa_versiones(tmp_path):
    registros = [
        ("a", "1", "SSD-A", "2026-10-05", 0.9, True),
        ("b", "1", "SSD-B", "2026-10-05", 0.1, False),
        ("c", "2", "SSD-A", "2026-10-05", 0.2, False),
        ("d", "2", "SSD-B", "2026-10-05", 0.8, True),
        ("e", "1", "SSD-C", "2026-11-18", 0.7, True),
        ("f", "1", "SSD-A", "2026-10-05", 0.95, True),  # Reintento del mismo SSD-día.
    ]
    ruta_registro = tmp_path / "predicciones.jsonl"
    with ruta_registro.open("w", encoding="utf-8") as archivo:
        for identificador, version, ssd, fecha, probabilidad, alerta in registros:
            archivo.write(json.dumps({
                "id_prediccion": identificador,
                "momento_utc": "2026-11-20T12:00:00+00:00",
                "version_modelo": version,
                "entrada": {"ssd_key": ssd, "observation_date": fecha},
                "salida": {"probabilidad_falla_30d": probabilidad,
                           "requiere_revision": alerta},
            }) + "\n")
    ruta_verdad = tmp_path / "verdad.csv"
    pd.DataFrame({
        CLAVE_SSD: ["SSD-A", "SSD-B", "SSD-C"],
        FECHA: ["2026-10-05", "2026-10-05", "2026-11-18"],
        OBJETIVO: [1, 0, 1],
    }).to_csv(ruta_verdad, index=False)

    predicciones = cargar_predicciones(ruta_registro)
    verdad = cargar_verdad(ruta_verdad)
    reporte = construir_reporte(predicciones, verdad, fecha_corte=date(2026, 11, 20))

    v1_octubre = reporte.loc[(reporte["version_modelo"] == "1") &
                            (reporte["periodo_inicio"] == "2026-10-05")].iloc[0]
    v2_octubre = reporte.loc[(reporte["version_modelo"] == "2") &
                            (reporte["periodo_inicio"] == "2026-10-05")].iloc[0]
    v1_noviembre = reporte.loc[reporte["periodo_inicio"] == "2026-11-16"].iloc[0]
    assert v1_octubre["pr_auc"] == 1.0
    assert v1_octubre["predicciones"] == 2
    assert v1_octubre["cobertura_verdad_madura"] == 1.0
    assert v2_octubre["pr_auc"] == 0.5
    assert v1_noviembre["verdades_disponibles"] == 0
    assert v1_noviembre["observaciones_maduras"] == 0
    assert pd.isna(v1_noviembre["pr_auc"])
    assert v1_noviembre["tasa_alertas"] == 1.0

    sin_verdad = construir_reporte(predicciones, None, fecha_corte=date(2026, 11, 20))
    assert sin_verdad["pr_auc"].isna().all()
    assert sin_verdad["predicciones"].sum() == 5
