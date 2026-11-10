#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 09 - Sesión 02 - Implementacion de: ejercicio1_fuga.py
#! -------------------------------------


#? Importar librerias
import os as os_environ
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline



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

def prepare_data_preprocessing_pipeline() -> Pipeline:
    """
    Crea un pipeline de preprocesamiento que incluye imputación con la mediana y escalado estándar.

    Satisface:
    # TODO 2: un Pipeline con tres pasos: imputar con la mediana, escalar y
    # la regresion lineal.

    :return: Pipeline de preprocesamiento.
    """
    return Pipeline([
        ("impute_median", SimpleImputer(strategy="median")),
        ("escala_standar", StandardScaler()),
        ("modelo_linear", LinearRegression())
    ])

def train_pipeline_and_evaluate(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Entrena el pipeline con los datos de entrenamiento y evalúa el rendimiento en los datos de prueba.

    Satisface:
    # TODO 3: el pipeline aprende solo con entrenamiento.

    :param pipeline: Pipeline de preprocesamiento y modelo.
    :param X_train: Conjunto de características de entrenamiento.
    :param y_train: Conjunto de etiquetas de entrenamiento.
    :param X_test: Conjunto de características de prueba.
    :param y_test: Conjunto de etiquetas de prueba.
    """

    #? Entrenamos solo con los datos de train, para evitar data leakage
    pipeline.fit(X_train, y_train)

    #? Calculamos metricas de salida
    mae = mean_absolute_error(y_test, pipeline.predict(X_test))
    mse = mean_squared_error(y_test, pipeline.predict(X_test))

    #? Imprimimos errores
    print("Errores de las pruebas realizadas".center(80, "="))
    print(f"Error absoluto medio en prueba: {mae:.4f}")
    print(f"Error cuadrático medio en prueba: {mse:.4f}")
    if pipeline is not None and "escala_standar" in pipeline.named_steps:
        escalador = pipeline.named_steps["escala_standar"]
        print("Parametros del escalador interno".center(80, "="))
        print("Filas que vio el escalador:", escalador.n_samples_seen_)
        print("Media de distancia usada para escalar:", round(escalador.mean_[0], 4))


def main() -> None:
    """
    Función principal que ejecuta el flujo completo de carga de datos, división, preprocesamiento y evaluación.
    """
    create_output_for_figures_directory()
    df = load_dataset_and_expose_loaded_object()
    if df is None:
        return

    #? Obtenemos los datos del dataset cargado.
    X = df[["distancia_km", "pasajeros"]]
    y = df["tarifa"]

    #? Realizamos la division correspondiente de los datos en entrenamiento y prueba.
    X_train, X_test, y_train, y_test = split_data_into_train_test(X, y)

    #? Obtenemos una instancia de la Pipeline para el escalado, imputacion y entrenamiento del modelo
    pipeline = prepare_data_preprocessing_pipeline()
    train_pipeline_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()