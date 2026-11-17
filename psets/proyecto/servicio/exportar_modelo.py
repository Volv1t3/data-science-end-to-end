"""Exporta la versión con alias produccion y su número para la imagen."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from modelado.entrenar_modelos import MODELO_REGISTRADO


SALIDA = Path(__file__).resolve().parent / "modelo.joblib"
VERSION = SALIDA.with_name("modelo_version.txt")


def exportar(salida: Path, version_archivo: Path, tracking_uri: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    version = MlflowClient().get_model_version_by_alias(MODELO_REGISTRADO, "produccion")
    corrida = mlflow.get_run(version.run_id)
    if corrida.info.status != "FINISHED":
        raise ValueError("La corrida de MLflow no terminó correctamente")
    modelo = mlflow.sklearn.load_model(f"models:/{MODELO_REGISTRADO}@produccion")
    columnas = getattr(modelo, "feature_names_in_", None)
    if columnas is None:
        raise ValueError("El pipeline exportado no conserva los nombres de sus features")
    paquete = {"modelo": modelo, "run_id": version.run_id,
               "version_modelo": str(version.version), "columnas": list(columnas), "umbral": 0.5}
    salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, salida)
    version_archivo.write_text(f"{version.version}\n", encoding="utf-8")
    print(f"Exportada versión {version.version} de {MODELO_REGISTRADO} a {salida}")
    return str(version.version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SALIDA)
    parser.add_argument("--version-file", type=Path, default=VERSION)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    args = parser.parse_args()
    exportar(args.output, args.version_file, args.tracking_uri)


if __name__ == "__main__":
    main()
