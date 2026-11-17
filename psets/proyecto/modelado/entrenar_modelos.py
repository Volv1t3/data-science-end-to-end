"""Reproduce las cuatro corridas MC1 del notebook en el MLflow local.

Ejecutar desde psets/proyecto: python -m modelado.entrenar_modelos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelado.pipeline_proyecto import (
    NUMERICAS,
    PARTICIONES,
    RUTA_DATOS,
    cargar_y_dividir,
)


PROYECTO = Path(__file__).resolve().parents[1]
MUESTRA_CURSO = PROYECTO.parents[1] / "laboratorios/semana-11-sesion-02/starter/mc1_muestra_experimentos.csv"
MANIFIESTO_CURSO = MUESTRA_CURSO.with_name("mc1_manifiesto_experimento.json")
EXPERIMENTO = "Antes de que falle - SSD MC1 - TARGET_30D"
MODELO_REGISTRADO = "antes_de_que_falle_mc1_clasificador"
SEMILLA = 42


def sha256_archivo(ruta: Path) -> str:
    huella = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            huella.update(bloque)
    return huella.hexdigest()


def candidatos() -> dict[str, Pipeline]:
    """Mismos nombres, estimadores e hiperparámetros del notebook original."""
    return {
        "linea_base_dummy": Pipeline([("modelo", DummyClassifier(strategy="most_frequent"))]),
        "bosque_aleatorio": Pipeline([("modelo", RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=SEMILLA, n_jobs=1,
        ))]),
        "gradient_boosting": Pipeline([("modelo", HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEMILLA,
        ))]),
        "red_neuronal_mlp": Pipeline([("escalar", StandardScaler()), ("modelo", MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", alpha=0.001,
            batch_size=256, learning_rate_init=0.001, max_iter=120,
            random_state=SEMILLA, early_stopping=False,
        ))]),
    }


def metricas(modelo: Pipeline, x: pd.DataFrame, y: pd.Series, prefijo: str) -> dict[str, float]:
    probabilidad = modelo.predict_proba(x)[:, 1]
    prediccion = (probabilidad >= 0.5).astype(int)
    precision_curva, recall_curva, _ = precision_recall_curve(y, probabilidad)
    promedio = float(average_precision_score(y, probabilidad))
    return {
        f"{prefijo}_pr_auc": promedio,
        f"{prefijo}_pr_auc_trapezoidal": float(auc(recall_curva, precision_curva)),
        f"{prefijo}_average_precision": promedio,
        f"{prefijo}_f05": float(fbeta_score(y, prediccion, beta=0.5, zero_division=0)),
        f"{prefijo}_precision": float(precision_score(y, prediccion, zero_division=0)),
        f"{prefijo}_recall": float(recall_score(y, prediccion, zero_division=0)),
    }


def entrenar(ruta: Path, uri: str, salida: Path, *, registrar: bool = True) -> dict:
    ruta = ruta.resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la muestra MC1: {ruta}")
    huella = sha256_archivo(ruta)
    manifiesto_fuente = MANIFIESTO_CURSO if ruta == MUESTRA_CURSO.resolve() else ruta.with_name(
        "mc1_muestra_experimentos.manifest.json"
    )
    if manifiesto_fuente.is_file():
        evidencia = json.loads(manifiesto_fuente.read_text(encoding="utf-8"))
        esperada = evidencia.get("muestra_sha256", evidencia.get("csv_sha256"))
        if esperada and esperada != huella:
            raise ValueError("La muestra no coincide con el SHA-256 del manifiesto")

    pares, columnas, auditoria = cargar_y_dividir(ruta)
    manifiesto = {
        "proyecto": EXPERIMENTO, "muestra_archivo": str(ruta),
        "muestra_sha256": huella, "caracteristicas_seleccionadas": columnas,
        "max_fraccion_nulos_entrenamiento": 0.05,
        "filas_finales": {p: len(pares[p][0]) for p in PARTICIONES},
        "semilla": SEMILLA,
        "versiones": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__,
                      "mlflow": mlflow.__version__, "plataforma": platform.platform()},
    }
    salida.mkdir(parents=True, exist_ok=True)
    ruta_manifiesto = salida / "mc1_entrenamiento.manifest.json"
    ruta_manifiesto.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    auditoria.to_csv(salida / "mc1_auditoria_caracteristicas.csv", index=False)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENTO)
    resultados = []
    for nombre, modelo in candidatos().items():
        with mlflow.start_run(run_name=nombre) as corrida:
            mlflow.set_tags({
                "candidato": "no", "proyecto": "SSD_MC1_TARGET_30D",
                "división": "cronológica_embargo_julio_octubre_2019",
                "muestra": "estratificada_entrenamiento", "modelo_familia": nombre,
            })
            mlflow.log_params({f"modelo__{k}": str(v)
                               for k, v in modelo.named_steps["modelo"].get_params(deep=False).items()})
            mlflow.log_params({
                "semilla": SEMILLA, "fecha_entrenamiento_fin": "2019-06-30",
                "fecha_validacion_inicio": "2019-08-01", "fecha_validacion_fin": "2019-09-30",
                "fecha_prueba_inicio": "2019-11-01", "fecha_prueba_fin": "2019-11-30",
                "embargos": "julio_y_octubre_2019", "datos_sha256": huella,
                "cantidad_features": len(columnas), "version_sklearn": sklearn.__version__,
                "version_mlflow": mlflow.__version__,
                **{f"filas_{p.lower()}": len(pares[p][0]) for p in PARTICIONES},
            })
            x_ent, y_ent = pares["ENTRENAMIENTO"]
            inicio = time.perf_counter()
            modelo.fit(x_ent, y_ent)
            resultados_run = {"segundos_entrenamiento": time.perf_counter() - inicio}
            for particion, prefijo in [("ENTRENAMIENTO", "entrenamiento"),
                                       ("VALIDACION", "validacion"), ("PRUEBA", "prueba")]:
                x, y = pares[particion]
                resultados_run.update(metricas(modelo, x, y, prefijo))
            mlflow.log_metrics(resultados_run)
            mlflow.log_artifact(str(ruta_manifiesto), artifact_path="procedencia")
            muestra = x_ent.head(5)
            info = mlflow.sklearn.log_model(
                sk_model=modelo, artifact_path="pipeline_entrenado",
                signature=infer_signature(muestra, modelo.predict(muestra)), input_example=muestra,
            )
            mlflow.set_tag("modelo_uri", info.model_uri)
            resultados.append({"modelo": nombre, "run_id": corrida.info.run_id,
                               "modelo_uri": info.model_uri, **resultados_run})
            print(nombre, corrida.info.run_id, f"PR-AUC validación={resultados_run['validacion_pr_auc']:.6f}", flush=True)

    tabla = pd.DataFrame(resultados).sort_values(
        ["validacion_pr_auc", "validacion_f05"], ascending=False, kind="stable"
    )
    tabla.to_csv(salida / "corridas_proyecto.csv", index=False)
    final = tabla.iloc[0]
    MlflowClient().set_tag(str(final["run_id"]), "candidato", "final")
    version = None
    if registrar:
        version = mlflow.register_model(model_uri=str(final["modelo_uri"]), name=MODELO_REGISTRADO)
    seleccion = {
        "experimento": EXPERIMENTO, "modelo": str(final["modelo"]),
        "run_id": str(final["run_id"]), "modelo_uri": str(final["modelo_uri"]),
        "version_registro": str(version.version) if version else None,
        "validacion_pr_auc": float(final["validacion_pr_auc"]),
        "prueba_pr_auc": float(final["prueba_pr_auc"]),
        "muestra_sha256": huella, "caracteristicas_seleccionadas": columnas,
    }
    (salida / "modelo_final.json").write_text(json.dumps(seleccion, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(seleccion, indent=2, ensure_ascii=False))
    return seleccion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RUTA_DATOS if RUTA_DATOS.exists() else MUESTRA_CURSO)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    parser.add_argument("--output-dir", type=Path, default=PROYECTO / "data/processed")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    entrenar(args.data, args.tracking_uri, args.output_dir, registrar=not args.no_register)


if __name__ == "__main__":
    main()
