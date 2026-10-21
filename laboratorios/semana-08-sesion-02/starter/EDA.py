#! -------------------------------------
#! Santiago Arellano 00328370
#! Semana 08 - Sesión 02 - Implementacion de: EDA base
#! -------------------------------------


#? Importar librerias
import os as os_environ
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_output_for_figures_directory() -> None:
    try:
        os_environ.makedirs("figures", exist_ok=True)
    except Exception as e:
        print(f"Error creating figures directory: {e}")


def load_dataset_and_expose_loaded_object() -> pd.DataFrame | None:
    try:
        df = pd.read_csv("./viajes_taxi.csv")
        return df
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return None

def explore_initial_dataset_summary_and_info(df: pd.DataFrame) -> None:
    print("Exploring initial dataset")
    print("Filas y columnas:", df.shape)
    print(f"Tipos de datos:\n{df.dtypes}")
    print(f"Valores nulos por columna:\n{df.isna().sum()}")
    print(f"Resumen estadístico:\n{df.describe().round(2)}")


def visualize_histogram_of_target_variable(df: pd.DataFrame, target_column: str, sbins: int =10) -> None:
    plt.figure()
    plt.hist(df[target_column], bins=sbins)
    plt.title(f"Histograma de {target_column} en base a {sbins} bins")
    plt.xlabel(target_column)
    plt.ylabel(f"Frecuencia de {target_column}")
    plt.savefig(f"figures/distribucion_{target_column}.png")
    plt.close()

def visualize_boxplot_of_target_variable(df: pd.DataFrame, target_column: str) -> None:
    plt.figure()
    sns.boxplot(x=df[target_column])
    plt.title(f"Boxplot de {target_column}")
    plt.xlabel(target_column)
    plt.ylabel(f"Valores de {target_column}")
    plt.savefig(f"figures/boxplot_{target_column}.png")
    plt.close()

def tabular_interquartile_analysis_of_target_variable(df: pd.DataFrame, target_column: str) -> None:
    #? Cuartiles para detectar atipicos
    q1 = df[target_column].quantile(0.25)
    q3 = df[target_column].quantile(0.75)
    iqr = q3 - q1

    #? Limites para detectar atipicos
    upper_lim = q3 + 1.5 * iqr
    lower_lim = q1 - 1.5 * iqr
    outliers = df[(df[target_column] < lower_lim) | (df[target_column] > upper_lim)]
    outliers_through_query = df.query(f"{target_column} < {lower_lim} or {target_column} > {upper_lim}")
    print(f"Atipicos de {target_column} segun IQR con df selection: {len(outliers)}")
    print(f"Atipicos de {target_column} segun IQR con query: {len(outliers_through_query)}")
    print(f"Primeros 10 atipicos:\n{outliers.head(10)}")
    print(f"Limites: {lower_lim} - {upper_lim}")
    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")


def visualize_two_col_scatterplot(df: pd.DataFrame, x_column: str, y_column: str) -> None:
    plt.figure()
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.title(f"Scatterplot de {y_column} vs {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.savefig(f"figures/scatter_{y_column}_vs_{x_column}.png")
    plt.close()

def visualize_correlation_heatmap(df: pd.DataFrame, name_variation: str="") -> None:
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa de calor de correlación")
    plt.savefig(f"figures/correlation_heatmap-{name_variation}.png")
    plt.close()

def visualize_correlation_heatmap_without_atypical(df: pd.DataFrame, name_variation: str="") -> None:

    #? Filter each column such that we only select those values within the range
    #? of 1.5 times the IQR from the first and third quartiles
    filtered_df = df.copy()
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_lim = q1 - 1.5 * iqr
        upper_lim = q3 + 1.5 * iqr
        filtered_df = filtered_df[(filtered_df[column] >= lower_lim) & (filtered_df[column] <= upper_lim)]

    #? Visualize the correlation heatmap of the filtered dataframe
    visualize_correlation_heatmap(filtered_df, name_variation)


def main():
    create_output_for_figures_directory()
    df = load_dataset_and_expose_loaded_object()
    if df is not None:
        explore_initial_dataset_summary_and_info(df)
        visualize_histogram_of_target_variable(df, target_column="tarifa", sbins=15)
        visualize_boxplot_of_target_variable(df, target_column="tarifa")
        tabular_interquartile_analysis_of_target_variable(df, target_column="tarifa")
        visualize_two_col_scatterplot(df, x_column="tarifa", y_column="distancia_km")
        visualize_correlation_heatmap(df)
        visualize_correlation_heatmap_without_atypical(df, name_variation="without_atypical")

    else:
        print("Dataset could not be loaded. Exiting.")

if __name__ == "__main__":
    main()
