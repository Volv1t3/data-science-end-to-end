#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 08 - Sesión 02 - Implementacion de: Feature Engineering por Zonas en Viajes Taxi
#! -------------------------------------

#? Importar librerias
import os as os_environ
from typing import Any
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as skpreprocessing

FILE_ROUTE= "./viajes_taxi.csv"
ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

def create_output_for_figures_directory() -> None:
    """
    Crea el directorio de salida de las figuras usando os.makedirs. Tolera si el directorio ya existe, e imprime un
    mensaje de error en caso de excepcion.
    """
    try:
        os_environ.makedirs("figures", exist_ok=True)
        os_environ.makedirs("figuras", exist_ok=True)
    except Exception as e:
        print(f"Error creating figures directory: {e}")


def load_dataset_and_expose_loaded_object(file_path:str = FILE_ROUTE) -> pd.DataFrame | None:
    """
    Intenta cargar el archivo de viajes de taxi y devuelve el DataFrame. Si no se encuentra el archivo,
    imprime un mensaje de error y devuelve None.
    """
    try:
        df = pd.read_csv(file_path)
        df = df[(df["tarifa"] != 999.99) & (df["duracion_min"] >= 0)].copy()
        return df
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return None

def create_tarifa_fija_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
    va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
    JFK Airport.

    Satisface:
    # TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
    # va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
    # JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
    # y maxima de esos viajes.

    :param df: DataFrame con los datos de viajes de taxi.
    :return: DataFrame con la nueva columna es_tarifa_fija.
    """
    df['es_tarifa_fija'] = np.where(
        ((df['zona_origen'] == 'JFK Airport') & (df['zona_destino'].isin(ZONAS_MANHATTAN))) |
        ((df['zona_destino'] == 'JFK Airport') & (df['zona_origen'].isin(ZONAS_MANHATTAN))),
        1, 0
    )

    # Imprime cuantos viajes la tienen en 1, y la tarifa minima y maxima de esos viajes.
    tarifa_fija_count = df['es_tarifa_fija'].sum()
    tarifa_fija_min = df[df['es_tarifa_fija'] == 1]['tarifa'].min()
    tarifa_fija_max = df[df['es_tarifa_fija'] == 1]['tarifa'].max()

    print(f"Cantidad de viajes con tarifa fija: {tarifa_fija_count}")
    print(f"Tarifa mínima de viajes con tarifa fija: {tarifa_fija_min}")
    print(f"Tarifa máxima de viajes con tarifa fija: {tarifa_fija_max}")

    return df

def encode_dummies_zona_origen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica la columna zona_origen con one-hot encoding.

    Satisface:
    # TODO 2: codifica zona_origen con one-hot (pd.get_dummies), con el
    # prefijo "origen" y dtype=int. Imprime los nombres de las columnas
    # nuevas.
    :param df: DataFrame con los datos de viajes de taxi.
    :return: DataFrame con las nuevas columnas de one-hot encoding.
    """
    dummies = pd.get_dummies(df['zona_origen'], prefix='origen', dtype=int)
    df = pd.concat([df, dummies], axis=1)

    # Imprime los nombres de las columnas nuevas.
    print("Nuevas columnas de one-hot encoding:")
    print(dummies.columns.tolist())

    return df

def encode_ordinal_zona_origen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica la columna zona_origen de forma ordinal.

    Satisface:
    # TODO 3: codifica zona_origen de forma ordinal: un numero entero por
    # zona, en orden alfabetico, en una columna origen_ordinal.
    :param df: DataFrame con los datos de viajes de taxi.
    :return: DataFrame con la nueva columna de codificación ordinal.
    """
    le = skpreprocessing.LabelEncoder()
    df['origen_ordinal'] = le.fit_transform(df['zona_origen'])

    return df

def check_columns_existence_before_trip(df: pd.DataFrame) -> None:
    """
    Para cada columna original del dataset, decide si existe en el momento en que alguien pide el taxi (antes del viaje)
    e imprime la decision.

    Satisface:
    # TODO 4: para cada columna original del dataset, decide si existe en el
    # momento en que alguien pide el taxi (antes del viaje) e imprime la
    # decision.
    :param df: DataFrame con los datos de viajes de taxi.
    """
    original_dataframe = load_dataset_and_expose_loaded_object(FILE_ROUTE)

    #? Para implmentar esta funcion, vamos a imprimir la decision por cada funcion en base al nombre de la columna
    print("Decisión sobre la existencia de columnas antes del viaje:")

    at_trip_init_columns = ['fecha_hora_inicio','zona_origen','zona_destino','pasajeros','metodo_pago']
    derived_columns = df.columns.difference(at_trip_init_columns)

    for column in original_dataframe.columns:
        if column in at_trip_init_columns:
            print(f"Columna '{column}': Existe antes del viaje (sí).")
        elif column in derived_columns:
            print(f"Columna '{column}': No existe antes del viaje (no).")

def export_final_dataset(df: pd.DataFrame, output_file_path: str = "viajes_taxi_final.csv") -> None:
    """
    Exporta el DataFrame final a un archivo CSV.

    :param df: DataFrame con los datos de viajes de taxi.
    :param output_file_path: Ruta del archivo de salida.
    """
    try:
        df.to_csv(output_file_path, index=False)
        print(f"Dataset final exportado a {output_file_path}")
    except Exception as e:
        print(f"Error exporting dataset: {e}")

def main():
    # Crear directorio de salida para figuras
    create_output_for_figures_directory()

    # Cargar el dataset
    df = load_dataset_and_expose_loaded_object(FILE_ROUTE)
    if df is None:
        return

    # Crear la columna es_tarifa_fija
    df = create_tarifa_fija_column(df)

    # Codificar zona_origen con one-hot encoding
    df = encode_dummies_zona_origen(df)

    # Codificar zona_origen de forma ordinal
    df = encode_ordinal_zona_origen(df)

    # Verificar la existencia de columnas antes del viaje
    check_columns_existence_before_trip(df)

    # Exportar el dataset final a un archivo CSV
    export_final_dataset(df, "viajes_taxi_final.csv")

if __name__ == "__main__":
    main()




