"""Extrae una muestra OBT MC1 reproducible mediante el conector Spark/Snowflake."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline_proyecto import CLAVE_SSD, FECHA, NUMERICAS, OBJETIVO, PARTICION, normalizar_tipos


MAX_FILAS_PERMITIDO = 50_000
IDENTIFICADOR = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def sha256_archivo(ruta: Path) -> str:
    huella = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            huella.update(bloque)
    return huella.hexdigest()


def nombre_obt() -> str:
    base = os.environ["SNOWFLAKE_DATABASE"].strip()
    esquema = os.getenv("SNOWFLAKE_OBT_SCHEMA", "S_CDATOS_PSET2_OBT").strip()
    if not IDENTIFICADOR.fullmatch(base) or not IDENTIFICADOR.fullmatch(esquema):
        raise ValueError("SNOWFLAKE_DATABASE y SNOWFLAKE_OBT_SCHEMA deben ser identificadores")
    return f'"{base}"."{esquema}"."OBT_MC1_RN"'


def construir_consulta(obt: str) -> str:
    columnas = ",\n       ".join(f"o.{columna}" for columna in NUMERICAS)
    return f"""
WITH candidatos AS (
    SELECT SSD_KEY, OBSERVATION_DATE, TARGET_30D,
           CASE
             WHEN OBSERVATION_DATE < '2019-07-01'::DATE THEN 'ENTRENAMIENTO'
             WHEN OBSERVATION_DATE BETWEEN '2019-08-01'::DATE AND '2019-09-30'::DATE THEN 'VALIDACION'
             WHEN OBSERVATION_DATE BETWEEN '2019-11-01'::DATE AND '2019-11-30'::DATE THEN 'PRUEBA'
           END AS PARTICION
    FROM {obt}
    WHERE TARGET_30D IN (0, 1)
      AND (OBSERVATION_DATE < '2019-07-01'::DATE
           OR OBSERVATION_DATE BETWEEN '2019-08-01'::DATE AND '2019-09-30'::DATE
           OR OBSERVATION_DATE BETWEEN '2019-11-01'::DATE AND '2019-11-30'::DATE)
), reducidos AS (
    SELECT * FROM candidatos
    WHERE PARTICION IS NOT NULL
      AND (TARGET_30D = 1 OR PARTICION IN ('VALIDACION', 'PRUEBA')
           OR MOD(MOD(HASH(SSD_KEY, OBSERVATION_DATE), 100) + 100, 100) < 5)
), numerados AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY PARTICION, TARGET_30D ORDER BY HASH(SSD_KEY, OBSERVATION_DATE), SSD_KEY, OBSERVATION_DATE) AS FILA_CLASE,
           ROW_NUMBER() OVER (PARTITION BY PARTICION ORDER BY HASH(SSD_KEY, OBSERVATION_DATE), SSD_KEY, OBSERVATION_DATE) AS FILA_PARTICION
    FROM reducidos
), seleccion AS (
    SELECT SSD_KEY, OBSERVATION_DATE, TARGET_30D, PARTICION
    FROM numerados
    WHERE (PARTICION = 'ENTRENAMIENTO'
           AND ((TARGET_30D = 1 AND FILA_CLASE <= 5000)
                OR (TARGET_30D = 0 AND FILA_CLASE <= 25000)))
       OR (PARTICION IN ('VALIDACION', 'PRUEBA') AND FILA_PARTICION <= 10000)
)
SELECT o.SSD_KEY, o.OBSERVATION_DATE,
       {columnas},
       s.TARGET_30D, s.PARTICION
FROM seleccion AS s
JOIN {obt} AS o ON o.SSD_KEY = s.SSD_KEY AND o.OBSERVATION_DATE = s.OBSERVATION_DATE
ORDER BY s.PARTICION, o.SSD_KEY, o.OBSERVATION_DATE
""".strip()


def consultar_snowflake(sql: str, *, max_filas: int) -> pd.DataFrame:
    from snowflake_io import create_spark_session, read_snowflake_query

    texto = sql.strip().rstrip(";")
    if not texto.upper().startswith(("SELECT", "WITH")) or ";" in texto:
        raise ValueError("La consulta debe ser una única sentencia SELECT o WITH")
    spark = create_spark_session("mc1-muestra-integracion")
    resultado = read_snowflake_query(
        spark,
        f"SELECT * FROM ({texto}) AS MUESTRA LIMIT {max_filas + 1}",
        schema=os.getenv("SNOWFLAKE_OBT_SCHEMA", "S_CDATOS_PSET2_OBT"),
    ).toPandas()
    if len(resultado) > max_filas:
        raise ValueError(f"La consulta devolvió más de {max_filas:,} filas")
    return resultado


def exportar_muestra(salida: Path, manifiesto: Path, *, max_filas: int) -> None:
    if not 1 <= max_filas <= MAX_FILAS_PERMITIDO:
        raise ValueError(f"max_filas debe estar entre 1 y {MAX_FILAS_PERMITIDO:,}")
    obt = nombre_obt()
    consulta = construir_consulta(obt)
    datos = normalizar_tipos(consultar_snowflake(consulta, max_filas=max_filas))
    if datos.empty:
        raise ValueError("Snowflake no devolvió filas elegibles para la muestra MC1")
    datos = datos.sort_values([PARTICION, CLAVE_SSD, FECHA], kind="stable").reset_index(drop=True)
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(salida, index=False, date_format="%Y-%m-%d", float_format="%.17g")
    evidencia = {
        "tabla_origen": obt,
        "filas": len(datos),
        "columnas": list(datos.columns),
        "csv_sha256": sha256_archivo(salida),
        "sql_sha256": hashlib.sha256(consulta.encode("utf-8")).hexdigest(),
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifiesto.parent.mkdir(parents=True, exist_ok=True)
    manifiesto.write_text(json.dumps(evidencia, indent=2), encoding="utf-8")
    print(f"Muestra MC1: {len(datos):,} filas en {salida}")


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=MAX_FILAS_PERMITIDO)
    return parser.parse_args()


if __name__ == "__main__":
    args = argumentos()
    exportar_muestra(args.output, args.manifest, max_filas=args.max_rows)
