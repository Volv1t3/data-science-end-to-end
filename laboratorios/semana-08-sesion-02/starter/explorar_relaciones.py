#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 08 - Sesión 02 - Implementacion de: Explorar Relaciones
#! -------------------------------------


#? Importar librerias
import os as os_environ
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

def helper_clean_single_column_outliers(df: pd.DataFrame, column_name: str, lower_bound: float | None, upper_bound: float | None) -> Any | None:
    """
    Funcion disenada para limpiar los valores atipicos de una columna especifica en un DataFrame.
    Un limite en None significa "sin limite" en ese lado (ej: duracion >= 0, sin tope superior).
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return None
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' does not exist in the DataFrame.")
        return None

    if lower_bound is not None and not isinstance(lower_bound, (int, float)):
        print("Error: Lower bound must be a numeric value or None.")
        return None
    if upper_bound is not None and not isinstance(upper_bound, (int, float)):
        print("Error: Upper bound must be a numeric value or None.")
        return None

    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        print("Error: Lower bound cannot be greater than upper bound.")
        return None

    if df[column_name].dtype not in [float, int]:
        print(f"Error: Column '{column_name}' must be numeric to filter outliers.")
        return None

    #? Ejecutamos logica real para filtrar los valores atipicos
    mask = pd.Series(True, index=df.index)
    if lower_bound is not None:
        mask &= df[column_name] >= lower_bound
    if upper_bound is not None:
        mask &= df[column_name] <= upper_bound
    cleaned_df = df[mask]
    return cleaned_df

def helper_remove_null_values_from_column(df: pd.DataFrame, column_name: str) -> Any | None:
    """
    Funcion disenada para eliminar los valores nulos de una columna especifica en un DataFrame
    """
    if df.empty or df is None:
        print("Error: The dataset file was not found.")
        return None
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' does not exist in the DataFrame.")
        return None

    #? Ejecutamos logica real para eliminar los valores nulos
    cleaned_df = df.dropna(subset=[column_name])
    return cleaned_df

def explore_initial_dataset_summary_and_info(df: pd.DataFrame) -> None:
    print("Exploring initial dataset")
    print("Filas y columnas:", df.shape)
    print(f"Tipos de datos:\n{df.dtypes}")
    print(f"Valores nulos por columna:\n{df.isna().sum()}")
    print(f"Resumen estadístico:\n{df.describe().round(2)}")

def plot_simple_scatterplot(df: pd.DataFrame, x_column: str, y_column: str, output_path: str) -> None:
    """
    Crea un gráfico de dispersión simple usando seaborn y lo guarda en la ruta especificada.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.title(f'Scatter plot of {y_column} vs {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.savefig(output_path)
    plt.close()

def calculate_tip_statistics(limpios: Any | None):
    propina_stats = limpios.groupby("metodo_pago").agg(
        propina_promedio=("propina", "mean"),
        cantidad_viajes_con_propina_0=("propina", lambda x: (x == 0).sum())
    ).reset_index()
    print("Propina promedio y cantidad de viajes con propina 0 por metodo de pago:")
    print(propina_stats)

def identify_anomalous_distance_fare_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica viajes sospechosos donde la distancia registrada
    es cero, pero existe una tarifa positiva.

    Devuelve un DataFrame con los registros identificados.
    """

    anomalous_mask = (
        (df["distancia_km"] == 0) &
        (df["tarifa"] > 0)
    )

    anomalous_records = df[anomalous_mask]

    print(
        f"Cantidad de viajes con distancia cero y tarifa positiva: "
        f"{len(anomalous_records)}"
    )

    return anomalous_records

def main():
    #? Crear directorio de salida para figuras
    create_output_for_figures_directory()

    #? Cargar dataset
    df = load_dataset_and_expose_loaded_object()
    if df is None:
        return

    #? Explorar dataset inicial
    explore_initial_dataset_summary_and_info(df)

    #? Calculamos los limites para tarifa que tiene el error de 999.99
    # ? Eliminamos solamente las tarifas erroneas de 999.99
    df = df[df["tarifa"] != 999.99]

    # ? Eliminamos valores de duracion negativa
    df = helper_clean_single_column_outliers(df, "duracion_min", 0, None)

    limpios = df
    print(f"\nCantidad de registros despues de limpieza: {len(limpios)}")

    # TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
    # (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
    # antes de seguir: busca puntos que no sigan la tendencia general y
    # cuentalos con un filtro.
    plot_simple_scatterplot(limpios, "distancia_km", "tarifa", "figuras/distancia_vs_tarifa.png")
    print()

    # TODO 2.1: Identificar relaciones sospechosas entre distancia y tarifa.
    anomalous_records = identify_anomalous_distance_fare_relationships(limpios)
    print(anomalous_records)

    # TODO 3: propina promedio y cantidad de viajes con propina 0, por
    # metodo de pago (groupby + agg).
    calculate_tip_statistics(limpios)

    # TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
    # con la tarifa.
    print("\nCorrelación de las variables con la tarifa:")
    print(limpios[["distancia_km", "duracion_min", "pasajeros", "propina"]].corrwith(limpios["tarifa"]).round(2))

    # TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
    pasajeros_vacios = limpios["pasajeros"].isna().sum()
    pasajeros_cero = (limpios["pasajeros"] == 0).sum()
    print(f"\nCantidad de viajes con pasajeros vacíos: {pasajeros_vacios}")
    print(f"Cantidad de viajes con pasajeros en 0: {pasajeros_cero}")

if __name__ == "__main__":
    main()


