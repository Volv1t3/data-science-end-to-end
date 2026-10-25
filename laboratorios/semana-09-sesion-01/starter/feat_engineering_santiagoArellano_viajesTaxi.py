#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 08 - Sesión 02 - Implementacion de: Feature Engineering
#! -------------------------------------


#? Importar librerias
import os as os_environ
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as skpreprocessing

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


def load_dataset_and_expose_loaded_object() -> pd.DataFrame | None:
    """
    Intenta cargar el archivo de viajes de taxi y devuelve el DataFrame. Si no se encuentra el archivo,
    imprime un mensaje de error y devuelve None.
    """
    try:
        df = pd.read_csv("./viajes_taxi.csv")
        return df
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return None


def perform_local_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza una limpieza local de los datos del DataFrame, eliminando valores atípicos y filas con valores nulos.
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return df

    # Eliminar valores atípicos en la columna 'duracion'
    df = df[(df['duracion_min'] >= 0) & (df["tarifa"] != 999.99)]

    return df


def extract_time_based_features(df: pd.DataFrame, date_column:str ) -> pd.DataFrame:
    """
    Extrae características basadas en el tiempo a partir de una columna de fecha y hora.
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return df

    if date_column not in df.columns:
        print(f"Error: Column '{date_column}' does not exist in the DataFrame.")
        return df

    # Convertir la columna de fecha a tipo datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Extraer características basadas en el tiempo
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['hour'] = df[date_column].dt.hour
    df['minute'] = df[date_column].dt.minute
    df['day_of_week'] = df[date_column].dt.dayofweek

    return df

def create_cyclic_sin_cos_varialbes(df: pd.DataFrame, columns_to_transform: list[str]) -> pd.DataFrame:
    """
    Crea variables cíclicas (seno y coseno) a partir de una columna numérica en el DataFrame.
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return df

    for column in columns_to_transform:
        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist in the DataFrame.")
            continue

        # Crear variables cíclicas
        df[f'{column}_sin'] = np.sin(2 * np.pi * (df[column] / 24))
        df[f'{column}_cos'] = np.cos(2 * np.pi * (df[column] / 24))

    return df

def create_dummy_variables_from_name_list(df: pd.DataFrame, columns_to_transform: list[str]) -> pd.DataFrame:
    """
    Crea variables dummy a partir de una columna categórica en el DataFrame, utilizando una lista de nombres.
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return df

    for column in columns_to_transform:
        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist in the DataFrame.")
            continue

        # Crear variables dummy
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)

    return df

def create_ordinal_encoding_from_name_list(df: pd.DataFrame, columns_to_transform: list[str]) -> pd.DataFrame:
    """
    Crea codificaciones ordinales a partir de una columna categórica en el DataFrame, utilizando una lista de nombres.
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return df

    for column in columns_to_transform:
        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist in the DataFrame.")
            continue

        # Crear codificación ordinal
        le = skpreprocessing.LabelEncoder()
        df[f'{column}_encoded'] = le.fit_transform(df[column])

    return df

def train_linear_releationship_model(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> Any:
    """
    Entrena un modelo de regresión lineal utilizando las columnas de características y la columna objetivo.
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return None

    from sklearn.linear_model import LinearRegression

    # Separar características y objetivo
    X = df[feature_columns]
    y = df[target_column]

    # Entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    return model

def main():
    # Crear directorio de salida para figuras
    create_output_for_figures_directory()

    # Cargar el dataset
    df = load_dataset_and_expose_loaded_object()
    if df is None:
        return

    # Realizar limpieza local
    df = perform_local_cleanup(df)

    # Extraer características basadas en el tiempo
    df = extract_time_based_features(df, 'fecha_hora_inicio')

    # Crear variables cíclicas para la columna 'hour'
    df = create_cyclic_sin_cos_varialbes(df, ['hour'])

    print(df)

    # Crear variables dummy para la columna 'zona_origen'
    df = create_dummy_variables_from_name_list(df, ['zona_origen'])

    print(df)

    # Crear variables ordinales para la columna metodo_pago
    df = create_ordinal_encoding_from_name_list(df, ['metodo_pago'])

    # Entrenar un modelo de regresión lineal utilizando las columnas de características y la columna objetivo
    # feature_columns = df.drop(columns=['fecha_hora_inicio', 'zona_origen','zona_destino','metodo_pago',
    #                                   'id_viaje','tarifa']).columns.tolist()
    # target_column = 'tarifa'
    # model = train_linear_releationship_model(df, feature_columns, target_column)


    # Guardar el DataFrame limpio y con nuevas características en un archivo CSV
    df.to_csv("viajes_taxi_cleaned.csv", index=False)
    # print("DataFrame cleaned and features extracted. Saved to 'viajes_taxi_cleaned.csv'.")
    # model.to_pickle("linear_regression_model.pkl")

if __name__ == "__main__":
    main()