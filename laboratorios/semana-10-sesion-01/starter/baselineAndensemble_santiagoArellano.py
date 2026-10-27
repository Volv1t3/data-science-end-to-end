#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 10 - Sesión 01 - Implementacion de: ejercicio1_linea_base_y_ensamblaje.py
#! -------------------------------------


#? Importar librerias
import os as os_environ
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


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


def preparacion(NUMERICAS: list[str], YA_LISTAS: list[str], CATEGORICAS: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
    ])




def evaluar(nombre, modelo, X_ent, y_ent, X_prueba, y_prueba, NUMERICAS, YA_LISTAS, CATEGORICAS):
    pipeline = Pipeline([("preparar", preparacion(NUMERICAS, YA_LISTAS, CATEGORICAS)), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    outcome = {
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    }
    print(f"{nombre:28s} entrenamiento {mae_ent:7.4f}   prueba {mae_prueba:7.4f}")
    return pipeline, outcome

def output_results_to_csv(resultados, filename="resultados_finales.csv"):
    """
    Guarda los resultados en un archivo CSV.
    """
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename}")

def main():
    create_output_for_figures_directory()
    df = load_dataset_and_expose_loaded_object()
    if df is None:
        return  # Exit if the dataset could not be loaded

    NUMERICAS = ["distancia_km", "pasajeros"]
    YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
    CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

    X = df[NUMERICAS + YA_LISTAS + CATEGORICAS]
    y = df["tarifa"]
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

    resultados = []

    #? 1. Dummy Regressor
    dummy_model = DummyRegressor(strategy="median")
    dummy_model.fit(X_ent, y_ent)
    pipe, outcome = evaluar("Dummy Regressor", dummy_model, X_ent, y_ent, X_prueba, y_prueba, NUMERICAS, YA_LISTAS, CATEGORICAS)
    resultados.append(outcome)

    #? 2. Decision Tree Regressor
    decision_tree_regressor = DecisionTreeRegressor(random_state=42)
    pipe, outcome = evaluar("Decision Tree Regressor", decision_tree_regressor, X_ent, y_ent, X_prueba, y_prueba, NUMERICAS, YA_LISTAS, CATEGORICAS)
    resultados.append(outcome)

    #? 3. Decision Tree Regressor with max_depth=8
    decision_tree_regressor = DecisionTreeRegressor(random_state=42, max_depth=8)
    pipe, outcome = evaluar("Decision Tree Regressor (max_depth=8)", decision_tree_regressor, X_ent, y_ent, X_prueba,
                            y_prueba, NUMERICAS, YA_LISTAS, CATEGORICAS)
    resultados.append(outcome)

    #? 4. Random Forest Regressor
    random_forest_regressor = RandomForestRegressor(random_state=42, n_estimators=200)
    pipe, outcome = evaluar("Random Forest Regressor", random_forest_regressor, X_ent, y_ent, X_prueba, y_prueba, NUMERICAS, YA_LISTAS, CATEGORICAS)
    resultados.append(outcome)

    #? 5. Print Results!!!
    resultados_df = pd.DataFrame(resultados)
    print("\nResultados finales:")
    print(resultados_df)
    output_results_to_csv(resultados_df)



if __name__  == "__main__":
    main()