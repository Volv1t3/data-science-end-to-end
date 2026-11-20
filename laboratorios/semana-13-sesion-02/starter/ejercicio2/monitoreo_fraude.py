"""Ejercicio 2: el monitoreo del servicio de fraude.

El servicio de fraude (version 2, el gradient boosting) guarda cada pago que
evalua en predicciones_fraude.csv: cuando llego, la probabilidad, si lo mando
a revision y cuanto tardo en responder. Los fraudes no se conocen en el
momento: se confirman cuando el titular de la tarjeta reclama el cobro (un
contracargo), entre 2 y 21 dias despues, y quedan en contracargos.csv.

Hoy es 14 de diciembre. El reporte separa las metricas del sistema (la
latencia), una metrica que se puede calcular el mismo dia (la tasa de
alertas) y la metrica del modelo que le importa al negocio (el recall), que
solo se puede calcular para las semanas cuyos fraudes ya se confirmaron.

    python monitoreo_fraude.py

Despues de resolver, corre python comprobar.py hasta que los cinco pasos
digan OK.
"""

import pandas as pd
from networkx.algorithms import dominance

HOY = "2026-12-14"
DEMORA_MAXIMA = 21          # dias que puede tardar en confirmarse un fraude
TASA_REFERENCIA = 0.0758    # pagos enviados a revision en la prueba, con el umbral 0.10
RECALL_REFERENCIA = 0.661   # recall en la prueba, con el umbral 0.10
LATENCIA_MAXIMA_MS = 50


def con_dia_y_semana(predicciones):
    """Agrega "dia" (la fecha) y "semana" (el lunes de la semana) a cada pago."""
    predicciones = predicciones.copy()
    fecha = pd.to_datetime(predicciones["fecha_hora"])
    #? Dia para la agrupacion requerida en T1
    predicciones["dia"] = fecha.dt.date
    #? Fecha inicial de la semana de la prediccion (al lunes como en el
    #? ejercicio 1
    predicciones["semana"] = fecha.dt.to_period("W").dt.start_time.dt.date
    return predicciones


def latencia_p95(predicciones: pd.DataFrame) -> pd.DataFrame:
    """La latencia p95 de cada dia, en milisegundos: una Series indexada por dia."""
    # [DONE] 1: agrupa por "dia" y calcula el cuantil 0.95 de "latencia_ms"
    # (.quantile(0.95)).

    #? Para agrupar esto, usamos la columna nueva de `dia` creada con la funcion de
    #? arriba. Como se pide una serie, entonces agrupamos en predicciones
    #? y extraemos la serie. El group by ya retorna una serie con el nombre de latencia_ms, el index
    #? de dia, y el cuartil por cada dia

    local_pred_grouping = (
        predicciones.groupby(by=['dia'])['latencia_ms'].quantile(0.95)
    )

    return local_pred_grouping


def tasa_de_alertas(predicciones: pd.DataFrame) -> pd.DataFrame:
    """La fraccion de pagos enviados a revision en cada semana: una Series
    indexada por semana."""
    # [DONE]  2: agrupa por "semana" y promedia la columna "revisar" (vale 1 o 0).

    #? En este caso hacemos lo mismo pero promediamos la columna revisar.
    #? En este caso, el promedio realizara la suma de todos los valores 0 y 1 y dividira
    #? para el total de valores, permitiendonos encontrar la fraccion directamente
    alerts_by_week_grouping = (
        predicciones.groupby(by=['semana'])['revisar'].mean()
    )

    return alerts_by_week_grouping



def fraudes_confirmados(contracargos: pd.DataFrame, hoy: str):
    """Los id_pago de los fraudes confirmados hasta hoy (incluido), como set."""
    # [DONE] 3: filtra los contracargos con fecha_confirmacion menor o igual a
    # hoy (las fechas son texto "AAAA-MM-DD", y se comparan bien como texto).

    #? Por cualquier problema convertimos la fecha a datetime como objeto sin hora
    contracargos['fecha_confirmacion'] = pd.to_datetime(contracargos["fecha_confirmacion"], errors='coerce')

    #? Utilizamos un fltrado directo en contracargados seleccionando en base a la serie
    #? booleana de la fecha <= hoy
    filtered_contracargos = contracargos[contracargos['fecha_confirmacion'] <= hoy]

    #? Obtenemos los ids y creamos el set
    return set(filtered_contracargos['id_pago'].values.tolist())


def recall_por_semana(predicciones: pd.DataFrame, contracargos: pd.DataFrame, hoy: str):
    """El recall de cada semana cuyos fraudes ya se confirmaron todos: una
    semana entra si su domingo, mas DEMORA_MAXIMA dias, es menor o igual a
    hoy. Devuelve un DataFrame indexado por semana, con las columnas
    "fraudes", "detectados" y "recall"."""
    # Marca como fraude los pagos que estan en fraudes_confirmados(...), deja
    # solo las semanas maduras, y por semana cuenta los fraudes, los fraudes
    # con revisar == 1, y su cociente.

    #? 1. Como en la anterior implementacion, ambos data frames requieren
    #? una coercion a date time por si acaso
    preds = predicciones.copy()
    preds['fecha_hora'] = pd.to_datetime(preds['fecha_hora'], errors='coerce')
    contra = contracargos.copy()
    contra['fecha_confirmacion'] = pd.to_datetime(contra['fecha_confirmacion'], errors='coerce')
    hoy_inner = pd.to_datetime(hoy).date()
    #? 2. Filtramos las semanas de predicciones en bas e a hoy y la demora maxima
    #? Esto evita contar a predicciones que fueron tomadas como revisar == 1 pero
    #? que los clientes no interactuaron con

    preds['domingo'] = preds['semana'] + pd.Timedelta(days=6 + DEMORA_MAXIMA)
    preds = preds[ preds['domingo'] <= hoy_inner]

    #? Creamos el resultado final idk
    fraudes = fraudes_confirmados(contracargos, hoy)

    preds["fraude"] = (
        preds["id_pago"].isin(fraudes)
    )

    preds["detectado"] = preds["fraude"] & (preds["revisar"] == 1)


    resultado = (
        preds
        .groupby("semana")
        .agg(
            fraudes=("fraude", "sum"),
            detectados=("detectado", "sum")
        )
    )
    resultado["recall"] = (
            resultado["detectados"]
            / resultado["fraudes"]
    )

    return resultado


def alertas(predicciones, contracargos, hoy):
    """Una lista de mensajes, uno por cada problema encontrado."""
    mensajes = []
    # Tres tipos de alerta, en este orden:
    # - "sistema: el <dia> la latencia p95 fue <x> ms", por cada dia con
    #   latencia p95 mayor que LATENCIA_MAXIMA_MS;
    # - "modelo (indirecta): la semana del <semana> mando a revision <x> de los pagos",
    #   por cada semana con una tasa de alertas menor que el 75% de TASA_REFERENCIA;
    # - "modelo: la semana del <semana> detecto <x> de los fraudes", por cada
    #   semana madura con recall menor que el 75% de RECALL_REFERENCIA.
    # Usa las funciones de arriba, y redondea x a 3 decimales (1 en la latencia).
    latencias = latencia_p95(predicciones)
    for dia, latencia in latencias[latencias > LATENCIA_MAXIMA_MS].items():
        mensajes.append(
            f"sistema: el {dia} la latencia p95 fue {latencia:.1f} ms"
        )

    tasas = tasa_de_alertas(predicciones)
    umbral_alertas = 0.75 * TASA_REFERENCIA
    for semana, tasa in tasas[tasas < umbral_alertas].items():
        mensajes.append(
            f"modelo (indirecta): la semana del {semana} mando a revision "
            f"{tasa:.3f} de los pagos"
        )

    recalls = recall_por_semana(predicciones, contracargos, hoy)
    umbral_recall = 0.75 * RECALL_REFERENCIA
    for semana, fila in recalls[recalls["recall"] < umbral_recall].iterrows():
        mensajes.append(
            f"modelo: la semana del {semana} detecto {fila['recall']:.3f} "
            f"de los fraudes"
        )

    return mensajes


if __name__ == "__main__":
    loaded_predictions = pd.read_csv("predicciones_fraude.csv")
    predicciones = con_dia_y_semana(loaded_predictions)
    contracargos = pd.read_csv("contracargos.csv")
    print(f"{len(predicciones)} pagos evaluados del {predicciones['dia'].min()} al {predicciones['dia'].max()}")
    print("\nLatencia p95 por dia (ms), los cinco dias mas lentos:")
    print(latencia_p95(predicciones).sort_values(ascending=False).head(5).round(1).to_string())
    print("\nTasa de alertas por semana:")
    print(tasa_de_alertas(predicciones).round(4).to_string())
    print(f"\nRecall por semana madura, al {HOY}:")
    print(recall_por_semana(predicciones, contracargos, HOY).to_string())
    print("\nAlertas:")
    for mensaje in alertas(predicciones, contracargos, HOY):
        print("-", mensaje)
