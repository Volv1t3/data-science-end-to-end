"""Ejercicio 2: el ciclo de vida del modelo de fraude.

El equipo entrena modelos candidatos de fraude cada tanto. Cada candidato
se registra como una version de "fraude-pagos", con su recall y su
precision al umbral de 0.10. Una version llega a "produccion" solo si
detecta mas fraudes (mayor recall) que la que esta ahi ahora. Cada
version queda etiquetada con el resultado de esa comparacion.

    python ciclo_fraude.py

Despues de resolver, corre python comprobar.py hasta que los cinco pasos
digan OK.
"""

import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from preparacion_pagos import agregar_features, columnas, preparacion

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "fraude-pagos"
ALIAS = "produccion"
UMBRAL = 0.10


def datos():
    pagos = agregar_features(pd.read_csv("pagos_taxi.csv"))
    return train_test_split(columnas(pagos), pagos["es_fraude"], test_size=0.2, random_state=42, stratify=pagos["es_fraude"])


def evaluar(modelo, X_prueba, y_prueba):
    alertas = modelo.predict_proba(X_prueba)[:, 1] >= UMBRAL
    return {"recall": recall_score(y_prueba, alertas), "precision": precision_score(y_prueba, alertas)}


def registrar(modelo, nombre_corrida, metricas):
    """Guarda el modelo y sus metricas en una corrida nueva, lo registra
    como version nueva de NOMBRE, y devuelve el numero de version."""
    # [DONE] 1: abre una corrida con run_name=nombre_corrida, guarda las
    # metricas (mlflow.log_metrics) y el modelo con
    # mlflow.sklearn.log_model(modelo, "modelo", registered_model_name=NOMBRE).
    # Devuelve el numero de version, como int.
    mlflow.set_experiment("fraude-pagos-ciclo")

    with mlflow.start_run(run_name=nombre_corrida) as run:
        #? Como el modelo se ingresa al sistema, solo guardamos el modelo
        #? con sus metricas evaluadas

        #? Registramos las metricas cargadas
        mlflow.log_metrics(metricas)

        #? Guardamos el modelo
        registered_model: ModelInfo= mlflow.sklearn.log_model(modelo,'modelo', registered_model_name=NOMBRE)

    return int(registered_model.registered_model_version)


def metrica_de(version, metrica):
    """La metrica guardada en la corrida que produjo esa version."""
    # [Done} 2: la version sabe de que corrida salio
    # (MlflowClient().get_model_version(NOMBRE, version).run_id), y la
    # corrida guarda sus metricas (get_run(run_id).data.metrics).

    #? Hasta donde entiendo, tengo que retornar las metricas de la version
    #? indicada en la llamada, y asumo que metrica es una string que indica
    #? que campo filtrar
    mlflow.set_tracking_uri("http://localhost:5050")

    cliente = MlflowClient()
    metadata_modelo: ModelVersion = cliente.get_model_version(name=NOMBRE, version=version)

    #? Del la metada usamos la informacon del run_id para obtener la ejecucion exacta
    ejecucion: Run =  cliente.get_run(run_id=metadata_modelo.run_id)

    #? Obtenemos las mtricas y retornamos la indiciada por metrica
    return ejecucion.data.metrics[metrica]



def version_en_produccion():
    """El numero de la version con el alias ALIAS, o None si ninguna lo tiene."""
    # [Done] 3: get_model_version_by_alias lanza MlflowException si el alias
    # no existe todavia (o si el modelo no existe).
    try:
        mlflow.set_tracking_uri("http://localhost:5050")

        cliente = MlflowClient()
        registrado = cliente.get_model_version_by_alias(NOMBRE, ALIAS)
        return int(registrado.version)
    except MlflowException as e:
        print("No se ha encontrado un modelo bajo el alias, ",ALIAS)
        return None


def promover_si_mejora(version):
    """Pone ALIAS en la version si no hay ninguna en produccion, o si su
    recall es mayor que el de la de produccion. Devuelve True si la promovio.
    En cualquier caso, etiqueta la version."""
    mlflow.set_tracking_uri("http://localhost:5050")
    cliente = MlflowClient()
    version_registro = str(version)

    # Capture the production version before changing the alias.  This is both
    # the version used for the recall comparison and the value of the
    # ``comparada_con`` tag.
    actual_version = version_en_produccion()
    comparada_con = "ninguna" if actual_version is None else str(actual_version)

    if actual_version is None:
        #? The first registered candidate becomes production automatically.
        mejora = True
    else:
        candidato = cliente.get_model_version(NOMBRE, version_registro)
        produccion = cliente.get_model_version(NOMBRE, str(actual_version))
        recall_candidato = cliente.get_run(candidato.run_id).data.metrics["recall"]
        recall_produccion = cliente.get_run(produccion.run_id).data.metrics["recall"]
        mejora = recall_candidato > recall_produccion

    if mejora:
        cliente.set_registered_model_alias(NOMBRE, ALIAS, version_registro)

    cliente.set_model_version_tag(
        NOMBRE, version_registro, "validacion", "aprobada" if mejora else "rechazada"
    )
    cliente.set_model_version_tag(NOMBRE, version_registro, "comparada_con", comparada_con)
    return mejora


if __name__ == "__main__":
    mlflow.set_experiment("fraude-pagos-ciclo")
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = datos()
    candidatos = [
        ("logistica", LogisticRegression(max_iter=1000)),
        ("gradient-boosting", GradientBoostingClassifier(random_state=42)),
        ("logistica-muy-regularizada", LogisticRegression(C=0.001, max_iter=1000)),
    ]
    for nombre, estimador in candidatos:
        modelo = Pipeline([("preparar", preparacion()), ("modelo", estimador)])
        modelo.fit(X_entrenamiento, y_entrenamiento)
        metricas = evaluar(modelo, X_prueba, y_prueba)
        version = registrar(modelo, nombre, metricas)
        promovida = promover_si_mejora(version)
        print(f"{nombre}: version {version}, recall {metricas['recall']:.3f}, precision {metricas['precision']:.3f}, promovida: {promovida}")
    print(f"En produccion: version {version_en_produccion()}")
