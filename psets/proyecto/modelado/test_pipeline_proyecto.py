"""Pruebas locales y autónomas para la preparación de datos MC1."""

from __future__ import annotations

import pandas as pd
import pytest

from modelado import pipeline_proyecto as pipeline


def _fila(ssd_key: str, fecha: str, objetivo: int) -> dict[str, object]:
    """Construye una fila OBT pequeña sin leer archivos ni usar credenciales."""
    valores: dict[str, object] = {
        pipeline.CLAVE_SSD: ssd_key,
        pipeline.FECHA: fecha,
        pipeline.OBJETIVO: objetivo,
        pipeline.PARTICION: pipeline.asignar_particion(
            pd.Series([fecha])
        ).iloc[0],
    }
    for indice, columna in enumerate(pipeline.NUMERICAS, start=1):
        valores[columna] = float(indice + objetivo)
    return valores


def _muestra_manual() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _fila("ssd-ent-1", "2019-06-25", 0),
            _fila("ssd-ent-2", "2019-06-26", 1),
            _fila("ssd-val-1", "2019-08-15", 0),
            _fila("ssd-prueba-1", "2019-11-15", 1),
        ]
    )


def test_objetivo_no_es_feature_del_modelo() -> None:
    datos = _muestra_manual()
    particiones = pipeline.dividir_por_fecha(datos)
    columnas = list(pipeline.NUMERICAS)

    pares_xy = pipeline.separar_xy(particiones, columnas)
    x_entrenamiento, y_entrenamiento = pares_xy["ENTRENAMIENTO"]

    assert pipeline.OBJETIVO not in x_entrenamiento.columns
    assert list(x_entrenamiento.columns) == columnas
    assert y_entrenamiento.tolist() == [0, 1]


def test_division_temporal_no_usa_filas_del_futuro() -> None:
    datos = _muestra_manual()
    embargo = _fila("ssd-embargo", "2019-10-10", 0)
    embargo[pipeline.PARTICION] = pd.NA
    datos = pd.concat([datos, pd.DataFrame([embargo])], ignore_index=True)

    particiones = pipeline.dividir_por_fecha(datos)

    assert particiones["ENTRENAMIENTO"][pipeline.FECHA].max() < pd.Timestamp(
        "2019-07-01"
    )
    assert particiones["VALIDACION"][pipeline.FECHA].min() >= pd.Timestamp(
        "2019-08-01"
    )
    assert particiones["PRUEBA"][pipeline.FECHA].min() >= pd.Timestamp("2019-11-01")
    assert "ssd-embargo" not in set(
        pd.concat(particiones.values())[pipeline.CLAVE_SSD]
    )


def test_seleccion_de_features_solo_aprende_del_entrenamiento() -> None:
    datos = _muestra_manual()
    particiones = pipeline.dividir_por_fecha(datos)
    columna = pipeline.NUMERICAS[0]

    particiones["ENTRENAMIENTO"][columna] = 1.0
    particiones["VALIDACION"][columna] = 999.0
    columnas, auditoria = pipeline.seleccionar_caracteristicas(
        particiones["ENTRENAMIENTO"]
    )

    assert columna not in columnas
    assert not bool(auditoria.loc[auditoria["caracteristica"] == columna, "utilizar"].iloc[0])


def test_rechaza_una_clave_ssd_dia_duplicada() -> None:
    datos = _muestra_manual()
    duplicado = pd.concat([datos, datos.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicados"):
        pipeline.normalizar_tipos(duplicado)
