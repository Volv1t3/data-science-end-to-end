"""Monitorea las predicciones MC1 por semana y versión del modelo.

Desde psets/proyecto:
    python -m monitoreo.desempeno
    python -m monitoreo.desempeno --verdad ../../laboratorios/semana-11-sesion-02/starter/mc1_muestra_experimentos.csv

La tasa de alertas se conoce al predecir. La PR-AUC solo se calcula para
observaciones con etiqueta disponible y con 30 días completos de seguimiento.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from modelado.pipeline_proyecto import CLAVE_SSD, FECHA, OBJETIVO


PROYECTO = Path(__file__).resolve().parents[1]
RUTA_REGISTRO = Path(__file__).resolve().parent / "predicciones.jsonl"
RUTA_REPORTE = PROYECTO / "data/processed/monitoreo_semanal.csv"


def cargar_predicciones(ruta: Path) -> pd.DataFrame:
    """Lee el JSONL del servicio y exige claves útiles para unir la verdad."""
    filas = []
    with ruta.open(encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, start=1):
            if not linea.strip():
                continue
            try:
                registro = json.loads(linea)
                entrada, salida = registro["entrada"], registro["salida"]
                probabilidad = float(salida["probabilidad_falla_30d"])
                if not math.isfinite(probabilidad) or not 0 <= probabilidad <= 1:
                    raise ValueError("probabilidad fuera de [0, 1]")
                alerta = salida["requiere_revision"]
                if not isinstance(alerta, bool):
                    raise ValueError("requiere_revision debe ser booleano")
                filas.append({
                    "id_prediccion": str(registro["id_prediccion"]),
                    "momento_utc": str(registro["momento_utc"]),
                    "version_modelo": str(registro["version_modelo"]),
                    "ssd_key": str(entrada["ssd_key"]),
                    "fecha_observacion": entrada["observation_date"],
                    "probabilidad_falla_30d": probabilidad,
                    "requiere_revision": alerta,
                })
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Registro inválido en la línea {numero}: {error}") from error
    if not filas:
        raise ValueError(f"No hay predicciones en {ruta}")
    predicciones = pd.DataFrame(filas)
    if predicciones["id_prediccion"].duplicated().any():
        raise ValueError("El registro contiene id_prediccion duplicados")
    predicciones["fecha_observacion"] = pd.to_datetime(
        predicciones["fecha_observacion"], errors="raise"
    ).dt.normalize()
    predicciones["momento_utc"] = pd.to_datetime(
        predicciones["momento_utc"], errors="raise", utc=True
    )
    return predicciones


def cargar_verdad(ruta: Path) -> pd.DataFrame:
    """Acepta la muestra histórica o un CSV de etiquetas finalizadas de la OBT."""
    verdad = pd.read_csv(
        ruta, usecols=[CLAVE_SSD, FECHA, OBJETIVO], dtype={CLAVE_SSD: "string"}
    )
    verdad[FECHA] = pd.to_datetime(verdad[FECHA], errors="raise").dt.normalize()
    verdad[OBJETIVO] = pd.to_numeric(verdad[OBJETIVO], errors="raise")
    if verdad.duplicated([CLAVE_SSD, FECHA]).any():
        raise ValueError("La verdad contiene pares SSD_KEY, OBSERVATION_DATE duplicados")
    if not verdad[OBJETIVO].dropna().isin([0, 1]).all():
        raise ValueError("TARGET_30D debe contener únicamente 0, 1 o nulos")
    return verdad


def construir_reporte(
    predicciones: pd.DataFrame,
    verdad: pd.DataFrame | None,
    *,
    fecha_corte: date,
    periodo: str = "semana",
) -> pd.DataFrame:
    """Calcula señales inmediatas y PR-AUC solo con verdad madura."""
    if periodo not in {"dia", "semana"}:
        raise ValueError("periodo debe ser dia o semana")
    # Los reintentos se conservan en el JSONL para auditoría. Una unidad SSD-día
    # aporta un solo caso a las métricas del modelo por versión: el más reciente.
    datos = (predicciones.sort_values("momento_utc", kind="stable")
             .drop_duplicates(["version_modelo", "ssd_key", "fecha_observacion"], keep="last")
             .copy())
    if verdad is None:
        datos[OBJETIVO] = pd.NA
    else:
        datos = datos.merge(
            verdad,
            how="left",
            left_on=["ssd_key", "fecha_observacion"],
            right_on=[CLAVE_SSD, FECHA],
            validate="many_to_one",
        )
    maduras = datos["fecha_observacion"] + pd.Timedelta(days=30) <= pd.Timestamp(fecha_corte)
    datos["horizonte_completo"] = maduras
    datos["verdad_disponible"] = maduras & datos[OBJETIVO].notna()
    datos["periodo_inicio"] = (
        datos["fecha_observacion"].dt.normalize()
        if periodo == "dia"
        else datos["fecha_observacion"].dt.to_period("W-SUN").dt.start_time
    )

    filas = []
    for (version, inicio), grupo in datos.groupby(["version_modelo", "periodo_inicio"], sort=True):
        evaluables = grupo.loc[grupo["verdad_disponible"]]
        etiquetas = evaluables[OBJETIVO].astype(int)
        pr_auc = (
            float(average_precision_score(etiquetas, evaluables["probabilidad_falla_30d"]))
            if etiquetas.nunique() == 2 else None
        )
        filas.append({
            "version_modelo": version,
            "periodo_inicio": inicio.date().isoformat(),
            "fecha_corte": fecha_corte.isoformat(),
            "predicciones": len(grupo),
            "tasa_alertas": float(grupo["requiere_revision"].mean()),
            "probabilidad_media": float(grupo["probabilidad_falla_30d"].mean()),
            "observaciones_maduras": int(grupo["horizonte_completo"].sum()),
            "verdades_disponibles": len(evaluables),
            "cobertura_verdad_madura": (
                len(evaluables) / int(grupo["horizonte_completo"].sum())
                if verdad is not None and grupo["horizonte_completo"].any() else None
            ),
            "positivos_confirmados": int(etiquetas.sum()),
            "pr_auc": pr_auc,
        })
    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registro", type=Path, default=RUTA_REGISTRO)
    parser.add_argument("--verdad", type=Path, help="CSV con SSD_KEY, OBSERVATION_DATE y TARGET_30D")
    parser.add_argument("--fecha-corte", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date())
    parser.add_argument("--periodo", choices=["dia", "semana"], default="semana")
    parser.add_argument("--salida", type=Path, default=RUTA_REPORTE)
    args = parser.parse_args()
    predicciones = cargar_predicciones(args.registro)
    verdad = cargar_verdad(args.verdad) if args.verdad else None
    reporte = construir_reporte(
        predicciones, verdad, fecha_corte=args.fecha_corte, periodo=args.periodo
    )
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    reporte.to_csv(args.salida, index=False)
    print(reporte.to_string(index=False))
    print(f"Reporte guardado en {args.salida}")


if __name__ == "__main__":
    main()
