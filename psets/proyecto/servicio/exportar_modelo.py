"""Exporta a joblib la corrida final registrada por entrenar_modelos.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


PROYECTO = Path(__file__).resolve().parents[1]
SELECCION = PROYECTO / "data/processed/modelo_final.json"
SALIDA = Path(__file__).resolve().parent / "modelo.joblib"


def exportar(run_id: str, salida: Path, tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    corrida = mlflow.get_run(run_id)
    if corrida.info.status != "FINISHED":
        raise ValueError("La corrida de MLflow no terminó correctamente")
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/pipeline_entrenado")
    columnas = getattr(modelo, "feature_names_in_", None)
    if columnas is None:
        raise ValueError("El pipeline exportado no conserva los nombres de sus features")
    paquete = {"modelo": modelo, "run_id": run_id, "columnas": list(columnas), "umbral": 0.5}
    salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, salida)
    print(f"Exportado {run_id} a {salida}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="ID explícito; por defecto usa modelo_final.json")
    parser.add_argument("--selection", type=Path, default=SELECCION)
    parser.add_argument("--output", type=Path, default=SALIDA)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    args = parser.parse_args()
    run_id = args.run_id or json.loads(args.selection.read_text(encoding="utf-8"))["run_id"]
    exportar(run_id, args.output, args.tracking_uri)


if __name__ == "__main__":
    main()
