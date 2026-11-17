"""Crea una petición real de prueba a partir de la muestra local MC1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modelado.pipeline_proyecto import CLAVE_SSD, FECHA, PARTICION


PROYECTO = Path(__file__).resolve().parents[1]
MUESTRA_CURSO = PROYECTO.parents[1] / "laboratorios/semana-11-sesion-02/starter/mc1_muestra_experimentos.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=MUESTRA_CURSO)
    parser.add_argument("--selection", type=Path, default=PROYECTO / "data/processed/modelo_final.json")
    parser.add_argument("--output", type=Path, default=PROYECTO / "data/processed/peticion_ejemplo.json")
    args = parser.parse_args()
    columnas = json.loads(args.selection.read_text(encoding="utf-8"))["caracteristicas_seleccionadas"]
    datos = pd.read_csv(args.data, dtype={CLAVE_SSD: "string"}, low_memory=False)
    elegibles = datos.loc[datos[PARTICION] == "PRUEBA"].dropna(subset=columnas)
    if elegibles.empty:
        raise ValueError("La muestra no contiene una observación de prueba completa")
    fila = elegibles.iloc[0]
    peticion = {"ssd_key": str(fila[CLAVE_SSD]), "observation_date": str(fila[FECHA]),
                "caracteristicas": {nombre: float(fila[nombre]) for nombre in columnas}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(peticion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
