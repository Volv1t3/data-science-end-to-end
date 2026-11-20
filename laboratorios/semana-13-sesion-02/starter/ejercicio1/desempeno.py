"""El desempeno del modelo en produccion: une cada prediccion guardada con
la tarifa que se cobro, y calcula el error por semana.

    python desempeno.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

MAE_REFERENCIA = 1.17  # el error en prueba con que se aprobo la version 1

if not os.path.exists("predicciones.jsonl"):
    sys.exit("No hay predicciones.jsonl: el servicio no esta guardando sus predicciones.")

predicciones: pd.DataFrame = pd.read_json("predicciones.jsonl", lines=True)
cobros: pd.DataFrame = pd.read_csv("cobros.csv")

# [DONE} 4: une predicciones y cobros por id_prediccion (pd.merge), calcula el
# error absoluto de cada viaje, agrega la columna "semana" con el lunes de
# la semana del viaje (fecha_hora_inicio.dt.to_period("W").dt.start_time.dt.date),
# y agrupa por semana: "viajes" (cuantos) y "mae" (el promedio del error).
por_semana: pd.DataFrame = pd.merge(predicciones, cobros, on="id_prediccion", how="inner")

#? Check 1: si por semana se creo, entonces tenemos que tener la misma
#? cantidad de records que predicciones solas, ya que usamos inner,
#? y las claves debeiran de ser unicas
if por_semana is not None:
    assert por_semana.shape[0] == predicciones.shape[0]
    assert por_semana.shape[0] == cobros.shape[0]
    assert por_semana['id_prediccion'].nunique() == predicciones.shape[0]
    assert por_semana['id_prediccion'].nunique() == cobros.shape[0]
print("Resultado Luego del Merge".center( 120, '='))
print(por_semana)
print("Resultado Luego del Merge".center( 120, '='))
print()

print("Resultado Luego del Error Absoluto".center(120, '='))
#* Una vez tenemos el merge, tenemos que calcular el error absolute de cada viaje
if por_semana is not None:
    por_semana['error_absoluto'] = np.abs(
        por_semana['tarifa_cobrada'] - por_semana['tarifa']
    )
print(por_semana)
print("Resultado Luego del Error Absoluto".center(120, '='))
print()

#? Agregamos la columan de la semana del viaje
if por_semana is not None:
    por_semana['fecha_hora_inicio'] = pd.to_datetime(por_semana['fecha_hora_inicio'], errors='coerce')
    por_semana['semana'] = por_semana['fecha_hora_inicio'].dt.to_period('W').dt.start_time.dt.date

print("Resultado Luego de Anadir la Semana".center(120, '='))
print(por_semana)
print("Resultado Luego de Anadir la Semana".center( 120, '='))

#? Finalmente agrupamos por semana
por_semana = (
    por_semana
    .groupby("semana")
    .agg(
        viajes=("id_prediccion", "count"),
        mae=("error_absoluto", "mean")
    )
    .reset_index()
)

por_semana["veces_la_referencia"] = (por_semana["mae"] / MAE_REFERENCIA).round(2)
print(por_semana)
