"""Mueve el alias produccion a una versión revisada del clasificador MC1."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from modelado.entrenar_modelos import MODELO_REGISTRADO


ALIAS = "produccion"
METRICA = "validacion_pr_auc"


def promover(version: str, tracking_uri: str, responsable: str, motivo: str, *, rollback: bool = False) -> None:
    if not responsable.strip() or not motivo.strip():
        raise ValueError("responsable y motivo son obligatorios para auditar la decisión")
    mlflow.set_tracking_uri(tracking_uri)
    cliente = MlflowClient()
    candidata = cliente.get_model_version(MODELO_REGISTRADO, version)
    corrida = cliente.get_run(candidata.run_id)
    if corrida.info.status != "FINISHED" or METRICA not in corrida.data.metrics:
        raise ValueError("La versión debe venir de una corrida finalizada con PR-AUC de validación")
    try:
        anterior = cliente.get_model_version_by_alias(MODELO_REGISTRADO, ALIAS)
    except MlflowException as error:
        # MLflow 2.17 devuelve INVALID_PARAMETER_VALUE cuando el alias aún no existe.
        if "alias produccion not found" not in str(error).lower():
            raise
        anterior = None
    previa = anterior.version if anterior else None
    if anterior and anterior.version != version and not rollback:
        metrica_anterior = cliente.get_run(anterior.run_id).data.metrics.get(METRICA)
        if metrica_anterior is not None and corrida.data.metrics[METRICA] < metrica_anterior:
            raise ValueError("La PR-AUC de validación cae frente a producción; use --rollback solo para revertir")

    cliente.set_registered_model_alias(MODELO_REGISTRADO, ALIAS, version)
    cliente.set_model_version_tag(MODELO_REGISTRADO, version, "promocion_responsable", responsable.strip())
    cliente.set_model_version_tag(MODELO_REGISTRADO, version, "promocion_motivo", motivo.strip())
    cliente.set_model_version_tag(MODELO_REGISTRADO, version, "promocion_fecha_utc", datetime.now(timezone.utc).isoformat())
    cliente.set_model_version_tag(MODELO_REGISTRADO, version, "promocion_desde_version", str(previa or "ninguna"))
    cliente.set_model_version_tag(MODELO_REGISTRADO, version, "promocion_pr_auc_validacion", str(corrida.data.metrics[METRICA]))
    print(f"{MODELO_REGISTRADO}@{ALIAS}: versión {version}, run {candidata.run_id}, PR-AUC={corrida.data.metrics[METRICA]:.6f}; anterior={previa}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Número de versión del registro")
    parser.add_argument("--responsable", required=True)
    parser.add_argument("--motivo", required=True)
    parser.add_argument("--rollback", action="store_true", help="Permite volver a una versión con menor métrica")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    args = parser.parse_args()
    promover(args.version, args.tracking_uri, args.responsable, args.motivo, rollback=args.rollback)


if __name__ == "__main__":
    main()
