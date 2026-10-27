#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 10 - Sesión 01 - Implementacion de: ejercicio2_regularizacion.py
#! -------------------------------------


# ? Importar librerías.
import os as os_environ
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]


def create_output_for_figures_directory() -> None:
    """Crea la carpeta de figuras si todavía no existe."""
    try:
        os_environ.makedirs("./figures", exist_ok=True)
    except OSError as error:
        print(f"Error creating figures directory: {error}")


def load_dataset_and_expose_loaded_object() -> pd.DataFrame | None:
    """Carga los viajes y devuelve None si el archivo no se encuentra."""
    try:
        return pd.read_csv("viajes_features.csv")
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return None


def dividir_datos(df: pd.DataFrame) -> tuple:
    """Separa las entradas y la tarifa antes de ajustar transformaciones."""
    X = df[NUMERICAS + YA_LISTAS + CATEGORICAS]
    y = df["tarifa"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preparacion(numericas: list[str], ya_listas: list[str], categoricas: list[str]) -> ColumnTransformer:
    """Construye la preparación sin ajustarla a ningún conjunto."""
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), numericas),
        ("ya_listas", "passthrough", ya_listas),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categoricas),
    ])


def crear_pipeline(modelo, grado_interacciones: int | None = None) -> Pipeline:
    """Agrega interacciones solo cuando el experimento las necesita."""
    pasos = [("preparar", preparacion(NUMERICAS, YA_LISTAS, CATEGORICAS))]
    if grado_interacciones is not None:
        pasos.append(("interacciones", PolynomialFeatures(
            degree=grado_interacciones, include_bias=False
        )))
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)


def evaluar(
        nombre: str,
        modelo,
        X_ent: pd.DataFrame,
        y_ent: pd.Series,
        X_prueba: pd.DataFrame,
        y_prueba: pd.Series,
        grado_interacciones: int | None = None,
) -> tuple[Pipeline, dict]:
    """Ajusta solo con entrenamiento y devuelve el pipeline y sus métricas."""
    pipeline = crear_pipeline(modelo, grado_interacciones)
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio

    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    resultado = {
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    }
    print(f"{nombre:34s} entrenamiento {mae_ent:7.4f}   prueba {mae_prueba:.4f}")
    return pipeline, resultado


def ejecutar_experimentos(X_ent, y_ent, X_prueba, y_prueba) -> tuple[list[dict], int]:
    """Ejecuta los cuatro modelos solicitados en los TODO 1 a 4."""
    resultados = []

    # ? TODO 1: línea base que siempre predice la mediana de las tarifas.
    _, resultado = evaluar(
        "linea base", DummyRegressor(strategy="median"),
        X_ent, y_ent, X_prueba, y_prueba
    )
    resultados.append(resultado)

    # ? TODO 2: regresión lineal con interacciones de grado 3.
    pipeline_lineal, resultado = evaluar(
        "lineal grado 3", LinearRegression(),
        X_ent, y_ent, X_prueba, y_prueba, grado_interacciones=3
    )
    resultados.append(resultado)
    columnas_grado_3 = pipeline_lineal.named_steps["interacciones"].n_output_features_
    print("Columnas generadas con grado 3:", columnas_grado_3)

    # ? TODO 3: regularizar la misma expansión polinómica con Ridge.
    _, resultado = evaluar(
        "ridge grado 3", Ridge(alpha=10.0),
        X_ent, y_ent, X_prueba, y_prueba, grado_interacciones=3
    )
    resultados.append(resultado)

    # ? Paso 4: ensamblaje boosting sin expansión polinómica.
    _, resultado = evaluar(
        "gradient boosting", GradientBoostingRegressor(random_state=42),
        X_ent, y_ent, X_prueba, y_prueba
    )
    resultados.append(resultado)
    return resultados, columnas_grado_3


def output_results_to_csv(resultados: list[dict], filename: str = "resultados_ejercicio2.csv") -> pd.DataFrame:
    """Guarda las métricas y devuelve la tabla para su revisión."""
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename}")
    return resultados_df


def comprobar(numero: int, condicion: bool) -> None:
    if condicion:
        print(f"Paso {numero}: OK")
    else:
        print(f"Paso {numero}: revisa")


def fila(resultados: list[dict], nombre: str) -> dict | None:
    for resultado in resultados:
        if resultado["modelo"] == nombre:
            return resultado
    return None


def main() -> None:
    create_output_for_figures_directory()
    df = load_dataset_and_expose_loaded_object()
    if df is None:
        return

    X_ent, X_prueba, y_ent, y_prueba = dividir_datos(df)
    resultados, columnas_grado_3 = ejecutar_experimentos(X_ent, y_ent, X_prueba, y_prueba)

    # ? Paso 5: exportar resultados y obtener el menor MAE de prueba.
    resultados_df = output_results_to_csv(resultados)
    mejor = resultados_df.loc[resultados_df["mae_prueba"].idxmin(), "modelo"]
    print("\nResultados finales:")
    print(resultados_df)
    print("Mejor modelo en prueba:", mejor)

    # ? Comprobaciones originales del ejercicio, sin cambiar sus condiciones.
    print()
    print("Comprobacion")
    comprobar(1,
              fila(resultados, "linea base") is not None and 12.0 < fila(resultados, "linea base")["mae_prueba"] < 13.0)
    comprobar(2, fila(resultados, "lineal grado 3") is not None and fila(resultados, "lineal grado 3")[
        "brecha"] > 100 and columnas_grado_3 == 3275)
    comprobar(3,
              fila(resultados, "ridge grado 3") is not None and fila(resultados, "ridge grado 3")["mae_prueba"] < 1.5)
    comprobar(4, fila(resultados, "gradient boosting") is not None and fila(resultados, "gradient boosting")[
        "mae_prueba"] < 1.35)
    comprobar(5, mejor == "gradient boosting" and os_environ.path.exists("resultados_ejercicio2.csv"))


if __name__ == "__main__":
    main()