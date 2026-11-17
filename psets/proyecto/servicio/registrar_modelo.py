"""Reentrena la MLP final MC1 y crea una versión nueva en MLflow.

Desde psets/proyecto: python -m servicio.registrar_modelo --alpha 0.01
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import sklearn
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, fbeta_score

from modelado.entrenar_modelos import (
    EXPERIMENTO,
    MANIFIESTO_CURSO,
    MODELO_REGISTRADO,
    MUESTRA_CURSO,
    sha256_archivo,
)
from modelado.pipeline_proyecto import RUTA_DATOS, cargar_y_dividir


def registrar(ruta: Path, tracking_uri: str, alpha: float) -> dict:
    """Ajusta solo con entrenamiento; valida antes de registrar, sin mover alias."""
    if alpha <= 0:
        raise ValueError("alpha debe ser mayor que cero")
    ruta = ruta.resolve()
    huella = sha256_archivo(ruta)
    manifiesto = MANIFIESTO_CURSO if ruta == MUESTRA_CURSO.resolve() else ruta.with_name(
        "mc1_muestra_experimentos.manifest.json"
    )
    if manifiesto.is_file():
        evidencia = json.loads(manifiesto.read_text(encoding="utf-8"))
        esperada = evidencia.get("muestra_sha256", evidencia.get("csv_sha256"))
        if esperada and esperada != huella:
            raise ValueError("La muestra no coincide con el SHA-256 del manifiesto")

    pares, columnas, _ = cargar_y_dividir(ruta)
    x_ent, y_ent = pares["ENTRENAMIENTO"]
    x_val, y_val = pares["VALIDACION"]

    # Misma arquitectura y semilla del notebook y de Semana 12; alpha permite
    # comparar versiones sin cambiar la preparación ni el corte temporal.
    from modelado.entrenar_modelos import candidatos

    modelo = candidatos()["red_neuronal_mlp"]
    modelo.set_params(modelo__alpha=alpha)
    modelo.fit(x_ent, y_ent)
    probabilidades = modelo.predict_proba(x_val)[:, 1]
    pr_auc = float(average_precision_score(y_val, probabilidades))
    f05 = float(fbeta_score(y_val, probabilidades >= 0.5, beta=0.5, zero_division=0))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="red_neuronal_mlp") as corrida:
        mlflow.set_tags({
            "proyecto": "SSD_MC1_TARGET_30D", "modelo_familia": "red_neuronal_mlp",
            "división": "cronológica_embargo_julio_octubre_2019",
            "muestra": "estratificada_entrenamiento", "versionado": "semana_13_sesion_01",
        })
        mlflow.log_params({
            "modelo__alpha": alpha, "datos_sha256": huella,
            "cantidad_features": len(columnas), "filas_entrenamiento": len(x_ent),
            "filas_validacion": len(x_val), "version_sklearn": sklearn.__version__,
        })
        mlflow.log_metrics({"validacion_pr_auc": pr_auc, "validacion_f05": f05})
        if manifiesto.is_file():
            mlflow.log_artifact(str(manifiesto), artifact_path="procedencia")
        ejemplo = x_ent.head(5).astype("float64")
        info = mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="pipeline_entrenado",
            registered_model_name=MODELO_REGISTRADO,
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
        )
        version = info.registered_model_version
        if version is None:
            raise RuntimeError("MLflow no devolvió el número de la versión registrada")
        resultado = {
            "modelo_registrado": MODELO_REGISTRADO,
            "version_modelo": str(version),
            "run_id": corrida.info.run_id,
            "alpha": alpha,
            "validacion_pr_auc": pr_auc,
            "validacion_f05": f05,
            "datos_sha256": huella,
        }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RUTA_DATOS if RUTA_DATOS.exists() else MUESTRA_CURSO)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    args = parser.parse_args()
    registrar(args.data, args.tracking_uri, args.alpha)


if __name__ == "__main__":
    main()
