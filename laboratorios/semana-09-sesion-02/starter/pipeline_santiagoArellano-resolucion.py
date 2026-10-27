#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 09 - Sesión 02 - Implementacion de: ejercicio2_pipeline.py
#! -------------------------------------


#? Importar librerias
import os as os_environ

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_output_for_figures_directory() -> None:
    """
    Crea el directorio de salida de las figuras usando os.makedirs. Tolera si el directorio ya existe, e imprime un
    mensaje de error en caso de excepcion.
    """
    try:
        os_environ.makedirs("figures", exist_ok=True)
    except Exception as e:
        print(f"Error creating figures directory: {e}")

def load_dataset_and_expose_loaded_object() -> pd.DataFrame | None:
    """
    Intenta cargar el archivo de viajes de taxi y devuelve el DataFrame. Si no se encuentra el archivo,
    imprime un mensaje de error y devuelve None.
    """
    try:
        df = pd.read_csv("./viajes_features.csv")
        return df
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return None


def split_data_into_train_test(X: pd.DataFrame, y: pd.Series, testSize: float = 0.2, randomState: int = 42,
                               shouldShuffle: bool=False) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento y prueba. Se configura los parametros indicados en el TODO como los
    parametros default para el metodo.

    Satisface:
    # TODO 1: divide X e y en entrenamiento (80%) y prueba (20%), con
    # random_state=42, antes de calcular cualquier estadistica.

    :param X: DataFrame de características.
    :param y: Serie de etiquetas.
    :param testSize: Porcentaje de datos para prueba.
    :param randomState: Semilla para la generación de números aleatorios.
    :param shouldShuffle: Si se debe barajar los datos antes de dividirlos.
    :return: Tupla con los conjuntos de entrenamiento y prueba.
    """
    return train_test_split(X, y, test_size=testSize, random_state=randomState, shuffle=shouldShuffle)

def prepare_numeric_variable_preprocessing_pipeline() -> Pipeline:
    """
    Crea un pipeline de preprocesamiento para variables numéricas. Este pipeline incluye un imputador que reemplaza
    valores faltantes con la mediana y un escalador estándar.

    Satisface:
    # TODO 2: arma un Pipeline para las columnas NUMERICAS (imputar con la
    # mediana, despues escalar)

    :return: Pipeline de preprocesamiento para variables numéricas.
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputar_con_mediana', SimpleImputer(strategy='median')),
        ('escalar_con_standar', StandardScaler())
    ])
    return numeric_pipeline

def prepare_categorical_variable_preprocessing_pipeline() -> Pipeline:
    """
    Crea un pipeline de preprocesamiento para variables categóricas. Este pipeline incluye un imputador que reemplaza
    valores faltantes con el valor más frecuente y un codificador OneHotEncoder.

    Satisface:
    # TODO 2: arma un Pipeline para las CATEGORICAS (imputar con el
    # valor mas frecuente, despues OneHotEncoder con handle_unknown="ignore" y
    # sparse_output=False)

    :return: Pipeline de preprocesamiento para variables categóricas.
    """
    categorical_pipeline = Pipeline(steps=[
        ('imputar_con_mas_frecuente', SimpleImputer(strategy='most_frequent')),
        ('codificar_con_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return categorical_pipeline

def orquestar_pipelines_de_preprocesamiento(numeric_pipeline: Pipeline, categorical_pipeline: Pipeline,
                                            numeric_variables: list[str], categorical_variables: list[str],
                                            already_ready_variables: list[str]
                                            ) -> ColumnTransformer:
    """
    Combina los pipelines de preprocesamiento para variables numéricas y categóricas en un ColumnTransformer.

    Satisface:
    # TODO 2: Juntalos en un ColumnTransformer llamado
    # preparacion, con las columnas YA_LISTAS pasando sin cambios
    # ("passthrough").

    :param numeric_pipeline: Pipeline de preprocesamiento para variables numéricas.
    :param categorical_pipeline: Pipeline de preprocesamiento para variables categóricas.
    :return: ColumnTransformer que combina ambos pipelines.
    """


    preparacion = ColumnTransformer(transformers=[
        ('numericas_con_numeric_pipe', numeric_pipeline, numeric_variables),
        ('categoricas_con_categorical_pipe', categorical_pipeline, categorical_variables),
        ('all_ready_con_passthrough', 'passthrough', already_ready_variables)
    ])

    return preparacion

def train_and_transform_data(preparacion: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Entrena el ColumnTransformer con los datos de entrenamiento y transforma tanto los datos de entrenamiento como
    los de prueba.

    Satisface:
    # TODO 3: aprende la preparacion solo con entrenamiento y transforma
    # entrenamiento y prueba. Imprime cuantas columnas quedan.

    :param preparacion: ColumnTransformer que combina ambos pipelines.
    :param X_train: Conjunto de características de entrenamiento.
    :param X_test: Conjunto de características de prueba.
    :return: Tupla con los conjuntos transformados de entrenamiento y prueba.
    """
    X_train_listo = preparacion.fit_transform(X_train)
    X_test_listo = preparacion.transform(X_test)

    print(f"Columnas después del preprocesamiento: {X_train_listo.shape[1]}")

    return X_train_listo, X_test_listo

def get_learned_medians(preparacion: ColumnTransformer) -> pd.Series:
    """
    Obtiene las medianas aprendidas por el imputador de variables numéricas.

    Satisface:
    # TODO 4: medianas_aprendidas, las medianas que aprendio el imputador de
    # numericas (atributo statistics_). Imprimelas junto a las medianas de
    # entrenamiento y las de todo el dataset.

    :param preparacion: ColumnTransformer que combina ambos pipelines.
    :return: Serie con las medianas aprendidas.
    """
    numeric_imputer = preparacion.named_transformers_['numericas_con_numeric_pipe'].named_steps['imputar_con_mediana']
    learned_medians = pd.Series(numeric_imputer.statistics_, index=preparacion.transformers_[0][2])

    print("Medianas aprendidas por el imputador:")
    print(learned_medians)

    return learned_medians

def get_medians_from_training_and_full_dataset(X_train: pd.DataFrame, X_full: pd.DataFrame, numeric_variables: list[str]) -> tuple:
    """
    Obtiene las medianas de las variables numéricas tanto del conjunto de entrenamiento como del conjunto completo.

    :param X_train: Conjunto de características de entrenamiento.
    :param X_full: Conjunto completo de características.
    :param numeric_variables: Lista de nombres de las variables numéricas.
    :return: Tupla con las medianas del conjunto de entrenamiento y del conjunto completo.
    """
    medians_training = X_train[numeric_variables].median()
    medians_full = X_full[numeric_variables].median()

    print("Medianas del conjunto de entrenamiento:")
    print(medians_training)
    print("Medianas del conjunto completo:")
    print(medians_full)

    return medians_training, medians_full

def transform_sample_data_using_pipelines(preparacion: ColumnTransformer, sample_data: pd.DataFrame) -> int:
    """
    Transforma un nuevo conjunto de datos utilizando el ColumnTransformer y cuenta cuántas columnas de zona_origen
    quedan en 1.

    Satisface:
    # TODO 5: transforma el viaje nuevo de abajo, que sale de una zona que el
    # entrenamiento nunca vio, y guarda en zona_origen_en_1 cuantas columnas
    # de zona_origen quedan en 1.

    :param preparacion: ColumnTransformer que combina ambos pipelines.
    :param sample_data: Nuevo conjunto de datos a transformar.
    :return: Número de columnas de zona_origen que quedan en 1.
    """
    transformed_sample: np.ndarray = preparacion.transform(sample_data)

    #? Realizamos la suma de todas las columnas que resultan del one hot encoding de zona_origen, buscando todas las
    #? columnas con esta string en su interior y realizamos la suma, dado que es un One Hot Encoding solo debebiramos
    # de sumar a 1
    zona_origen_en_1 = 0
    for value, column in zip(*transformed_sample.tolist(), preparacion.get_feature_names_out()):
        print(f"[DEBUG]\t Column Name: {column}: {value}")
        if "zona_origen" in column and value == 1:
            zona_origen_en_1 += 1

    return zona_origen_en_1

def main() -> None:
    """
    Función principal que ejecuta el flujo completo de carga de datos, división, preprocesamiento y evaluación.
    """
    create_output_for_figures_directory()
    df = load_dataset_and_expose_loaded_object()
    if df is None:
        return

    #? Obtenemos los datos del dataset cargado.
    NUMERICAS = ["distancia_km", "pasajeros"]
    YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
    CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

    X = df[NUMERICAS + YA_LISTAS + CATEGORICAS]
    y = df["tarifa"]

    #? Realizamos la division correspondiente de los datos en entrenamiento y prueba.
    X_ent, X_prueba, y_ent, y_prueba = split_data_into_train_test(X, y)

    #? Creamos los pipelines de preprocesamiento para variables numericas y categoricas
    numeric_pipeline = prepare_numeric_variable_preprocessing_pipeline()
    categorical_pipeline = prepare_categorical_variable_preprocessing_pipeline()

    #? Orquestamos los pipelines en un ColumnTransformer
    preparacion = orquestar_pipelines_de_preprocesamiento(numeric_pipeline, categorical_pipeline,
                                                          NUMERICAS, CATEGORICAS, YA_LISTAS)

    #? Entrenamos y transformamos los datos
    X_ent_listo, X_prueba_listo = train_and_transform_data(preparacion, X_ent, X_prueba)

    #? Obtenemos las medianas aprendidas por el imputador de variables numericas
    medianas_aprendidas = get_learned_medians(preparacion)

    #? Obtenemos las medianas del conjunto de entrenamiento y del conjunto completo
    medians_training, medians_full = get_medians_from_training_and_full_dataset(X_ent, X, NUMERICAS)

    #? Transformamos un nuevo conjunto de datos y contamos cuantas columnas de zona_origen quedan en 1
    nuevo = pd.DataFrame([{
        "distancia_km": 9.0, "pasajeros": 1, "hora_sin": 0.5, "hora_cos": -0.866,
        "es_fin_de_semana": 0, "es_hora_pico": 1, "es_tarifa_fija": 0,
        "zona_origen": "Bronx", "zona_destino": "Midtown", "metodo_pago": "tarjeta",
    }])
    zona_origen_en_1 = transform_sample_data_using_pipelines(preparacion, nuevo)

    def comprobar(numero, condicion):
        if condicion:
            print(f"TODO {numero}: OK")
        else:
            print(f"TODO {numero}: revisa")

    print()
    print("Comprobacion")
    comprobar(1, X_ent is not None and len(X_ent) == 2380 and len(X_prueba) == 595)
    comprobar(2, isinstance(preparacion, ColumnTransformer) and len(preparacion.transformers) == 3)
    comprobar(3, X_ent_listo is not None and X_ent_listo.shape[1] == 25 and X_prueba_listo.shape[0] == 595)
    comprobar(4, medianas_aprendidas is not None and X_ent is not None
              and np.allclose(medianas_aprendidas, X_ent[NUMERICAS].median().values))
    comprobar(5, zona_origen_en_1 == 0)

if __name__ == "__main__":
    main()