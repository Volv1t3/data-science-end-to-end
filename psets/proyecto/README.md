# [Nombre de tu proyecto]

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

[Un párrafo: qué problema de negocio resuelve, para quién, y qué predice el modelo. Sale de tu propuesta del PSet 1, que está completa en `docs/`.]

**Métrica de éxito:** [la métrica de negocio y el umbral de decisión del PSet 1]

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| [nombre de la fuente] | [qué filas y columnas trae, desde qué fecha] | [la URL, la API o el procedimiento exacto] | `data/raw/[archivo]` |

Los datos no están en el repositorio. Para regenerarlos desde cero: [los comandos, en orden]. Se completa con el PSet 1.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

La muestra de entrenamiento se extrae de la OBT MC1 ya construida en Snowflake.
En el entorno Spark del PSet 2, ejecutar desde `modelado/`:

```bash
python exportar_muestra_mc1.py \
  --output ../data/processed/mc1_muestra_experimentos.csv \
  --manifest ../data/processed/mc1_muestra_experimentos.manifest.json
```

El archivo local contiene hasta 50 000 observaciones SSD-día de
`OBT_MC1_RN`, con 54 features temporales de SMART y la división cronológica
de entrenamiento, validación y prueba. No se sube a Git.

Después, `pipeline_proyecto.py` valida el CSV, selecciona features usando
solo entrenamiento y prepara los tres conjuntos sin conectarse a Snowflake:

```bash
cd modelado
python pipeline_proyecto.py
```

Las pruebas del módulo son independientes del CSV y de las credenciales. Desde
la raíz del repositorio:

```bash
pytest psets/proyecto -v
```

El workflow manual `MC1 Snowflake integration` construye la imagen Spark del
PSet 2 y valida una extracción real. Requiere el Environment
`snowflake-integration` y sus secrets/variables; no se ejecuta en cada push.

## Modelo final

[El modelo elegido, su `run_id` en MLflow y la métrica que lo sostiene. El razonamiento completo está en `docs/decision_modelo.md`. Se completa con el PSet 3.]

## Cómo correr el servicio

[Cómo se construye y se levanta el contenedor de `servicio/`, qué versión del modelo carga desde el registro, y un ejemplo de petición con su respuesta. Se completa con el PSet 4.]
