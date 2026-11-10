"""Verifica el contrato de una muestra MC1 creada por exportar_muestra_mc1.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from exportar_muestra_mc1 import MAX_FILAS_PERMITIDO, sha256_archivo
from pipeline_proyecto import CLAVE_SSD, FECHA, NUMERICAS, OBJETIVO, PARTICION, PARTICIONES, asignar_particion, normalizar_tipos


COLUMNAS_ESPERADAS = [CLAVE_SSD, FECHA, *NUMERICAS, OBJETIVO, PARTICION]


def verificar_muestra(csv: Path, manifiesto: Path) -> None:
    if not csv.is_file() or not manifiesto.is_file():
        raise FileNotFoundError("Se requieren el CSV de muestra y su manifiesto")
    datos = pd.read_csv(csv, dtype={CLAVE_SSD: "string"}, low_memory=False)
    if datos.empty:
        raise ValueError("La muestra está vacía")
    if len(datos) > MAX_FILAS_PERMITIDO:
        raise ValueError("La muestra supera el límite de seguridad")
    if list(datos.columns) != COLUMNAS_ESPERADAS:
        raise ValueError("Las columnas u orden del CSV no coinciden con el contrato")
    datos = normalizar_tipos(datos)
    if not datos[PARTICION].equals(asignar_particion(datos[FECHA])):
        raise ValueError("PARTICION no coincide con OBSERVATION_DATE")
    ausentes = set(PARTICIONES).difference(datos[PARTICION].unique())
    if ausentes:
        raise ValueError(f"La muestra no contiene particiones: {sorted(ausentes)}")
    evidencia = json.loads(manifiesto.read_text(encoding="utf-8"))
    if evidencia.get("csv_sha256") != sha256_archivo(csv):
        raise ValueError("La huella del CSV no coincide con el manifiesto")
    if evidencia.get("filas") != len(datos):
        raise ValueError("El conteo de filas no coincide con el manifiesto")
    print(f"Contrato MC1 válido: {len(datos):,} filas y {len(datos.columns)} columnas")


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = argumentos()
    verificar_muestra(args.csv, args.manifest)
