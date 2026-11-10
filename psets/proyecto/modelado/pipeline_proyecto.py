"""Preparación reproducible para el clasificador de fallas SSD MC1.

Este módulo recibe una muestra ya extraída de ``OBT_MC1_RN``.  No abre una
sesión Spark ni consulta Snowflake: esas responsabilidades viven en
``exportar_muestra_mc1.py`` para que el modelado y sus pruebas sean locales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OBJETIVO = "TARGET_30D"
CLAVE_SSD = "SSD_KEY"
FECHA = "OBSERVATION_DATE"
PARTICION = "PARTICION"
PARTICIONES = ("ENTRENAMIENTO", "VALIDACION", "PRUEBA")

SMART = ("R_1", "N_1", "R_5", "N_5", "R_9", "N_9")
VENTANAS = (7, 14, 30)
TIPOS = ("MEAN", "COUNT", "MEAN_SHIFT")
NUMERICAS = tuple(
    f"{smart}_{tipo}_{ventana}D"
    for smart in SMART
    for ventana in VENTANAS
    for tipo in TIPOS
)

RUTA_DATOS = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "mc1_muestra_experimentos.csv"
)


def asignar_particion(fechas: pd.Series) -> pd.Series:
    """Devuelve la partición temporal o ``NA`` para los meses de embargo."""
    fechas = pd.to_datetime(fechas, errors="raise")
    particiones = pd.Series(pd.NA, index=fechas.index, dtype="string")
    particiones.loc[fechas < pd.Timestamp("2019-07-01")] = "ENTRENAMIENTO"
    particiones.loc[
        fechas.between(pd.Timestamp("2019-08-01"), pd.Timestamp("2019-09-30"))
    ] = "VALIDACION"
    particiones.loc[
        fechas.between(pd.Timestamp("2019-11-01"), pd.Timestamp("2019-11-30"))
    ] = "PRUEBA"
    return particiones


def normalizar_tipos(datos: pd.DataFrame) -> pd.DataFrame:
    """Valida el contrato de la muestra OBT sin imputar mediciones SMART."""
    requeridas = [CLAVE_SSD, FECHA, *NUMERICAS, OBJETIVO, PARTICION]
    faltantes = sorted(set(requeridas).difference(datos.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    resultado = datos.copy()
    resultado[CLAVE_SSD] = resultado[CLAVE_SSD].astype("string")
    resultado[FECHA] = pd.to_datetime(resultado[FECHA], errors="raise")
    resultado[list(NUMERICAS)] = resultado[list(NUMERICAS)].apply(
        pd.to_numeric, errors="coerce"
    )
    resultado[list(NUMERICAS)] = resultado[list(NUMERICAS)].replace(
        [np.inf, -np.inf], np.nan
    )
    resultado[OBJETIVO] = pd.to_numeric(resultado[OBJETIVO], errors="raise").astype(
        "int64"
    )
    resultado[PARTICION] = resultado[PARTICION].astype("string")

    if not set(resultado[OBJETIVO].unique()).issubset({0, 1}):
        raise ValueError("TARGET_30D debe contener solamente 0 o 1")
    if not set(resultado[PARTICION].dropna().unique()).issubset(PARTICIONES):
        raise ValueError("La muestra contiene una partición desconocida")
    if resultado.duplicated([CLAVE_SSD, FECHA]).any():
        raise ValueError("La muestra contiene pares SSD_KEY, OBSERVATION_DATE duplicados")
    return resultado


def dividir_por_fecha(datos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Divide cronológicamente y rechaza etiquetas que no concuerdan con la fecha."""
    datos = normalizar_tipos(datos)
    esperadas = asignar_particion(datos[FECHA])
    if not datos[PARTICION].equals(esperadas):
        raise ValueError("PARTICION no coincide con los cortes temporales del proyecto")

    resultado = {
        nombre: datos.loc[datos[PARTICION] == nombre].copy()
        for nombre in PARTICIONES
    }
    if any(conjunto.empty for conjunto in resultado.values()):
        raise ValueError("La muestra debe incluir entrenamiento, validación y prueba")
    return resultado


def seleccionar_caracteristicas(
    entrenamiento: pd.DataFrame, *, max_fraccion_nulos: float = 0.05
) -> tuple[list[str], pd.DataFrame]:
    """Decide features solo con entrenamiento; validación y prueba no influyen."""
    if not 0 <= max_fraccion_nulos <= 1:
        raise ValueError("max_fraccion_nulos debe estar entre 0 y 1")
    faltantes = sorted(set(NUMERICAS).difference(entrenamiento.columns))
    if faltantes:
        raise ValueError(f"Faltan features de entrenamiento: {faltantes}")

    auditoria = pd.DataFrame(
        {
            "caracteristica": NUMERICAS,
            "fraccion_nulos_entrenamiento": entrenamiento[list(NUMERICAS)]
            .isna()
            .mean()
            .to_numpy(),
            "valores_distintos_entrenamiento": entrenamiento[list(NUMERICAS)]
            .nunique(dropna=True)
            .to_numpy(),
        }
    )
    auditoria["utilizar"] = (
        auditoria["fraccion_nulos_entrenamiento"] <= max_fraccion_nulos
    ) & (auditoria["valores_distintos_entrenamiento"] > 1)
    columnas = auditoria.loc[auditoria["utilizar"], "caracteristica"].tolist()
    if len(columnas) < 2:
        raise ValueError("Quedaron menos de dos características utilizables")
    return columnas, auditoria


def preparar_conjuntos_sin_imputacion(
    particiones: dict[str, pd.DataFrame], columnas: list[str]
) -> dict[str, pd.DataFrame]:
    """Elimina filas incompletas sin inventar lecturas SMART faltantes."""
    faltantes_particiones = set(PARTICIONES).difference(particiones)
    if faltantes_particiones:
        raise ValueError(f"Faltan particiones: {sorted(faltantes_particiones)}")

    preparados: dict[str, pd.DataFrame] = {}
    for nombre in PARTICIONES:
        limpio = particiones[nombre].dropna(subset=columnas).copy()
        if limpio.empty:
            raise ValueError(f"La limpieza eliminó todas las filas de {nombre}")
        preparados[nombre] = limpio
    return preparados


def separar_xy(
    particiones: dict[str, pd.DataFrame], columnas: list[str]
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Separa X/y y garantiza que ni objetivo ni columnas futuras entren a X."""
    columnas_prohibidas = {
        OBJETIVO,
        CLAVE_SSD,
        FECHA,
        PARTICION,
        "FIRST_FAILURE_DATE",
        "LAST_SEEN_DATE",
        "DAYS_TO_FAILURE",
        "LABEL_STATUS",
    }
    if set(columnas).intersection(columnas_prohibidas):
        raise ValueError("La lista de features contiene objetivo, identidad o información futura")
    return {
        nombre: (datos.loc[:, columnas].copy(), datos[OBJETIVO].copy())
        for nombre, datos in particiones.items()
    }


def construir_preparacion(columnas: list[str]) -> ColumnTransformer:
    """Crea el escalado que se ajustará exclusivamente con entrenamiento."""
    if not columnas:
        raise ValueError("Se requiere al menos una característica")
    return ColumnTransformer(
        [("numericas", Pipeline([("escalar", StandardScaler())]), columnas)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def cargar_y_dividir(
    ruta: str | Path = RUTA_DATOS, *, max_fraccion_nulos: float = 0.05
) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], list[str], pd.DataFrame]:
    """Carga la muestra local y devuelve pares X/y, features y auditoría."""
    datos = pd.read_csv(ruta, dtype={CLAVE_SSD: "string"}, low_memory=False)
    particiones = dividir_por_fecha(datos)
    columnas, auditoria = seleccionar_caracteristicas(
        particiones["ENTRENAMIENTO"], max_fraccion_nulos=max_fraccion_nulos
    )
    preparadas = preparar_conjuntos_sin_imputacion(particiones, columnas)
    return separar_xy(preparadas, columnas), columnas, auditoria


if __name__ == "__main__":
    pares_xy, columnas, _ = cargar_y_dividir()
    preparacion = construir_preparacion(columnas)
    x_entrenamiento, _ = pares_xy["ENTRENAMIENTO"]
    x_prueba, _ = pares_xy["PRUEBA"]
    x_entrenamiento_listo = preparacion.fit_transform(x_entrenamiento)
    x_prueba_lista = preparacion.transform(x_prueba)
    print("Features seleccionadas:", len(columnas))
    print("Entrenamiento:", x_entrenamiento_listo.shape)
    print("Prueba:", x_prueba_lista.shape)
