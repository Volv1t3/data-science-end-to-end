# Antes de que falle — SSD MC1

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

Para repetir las cuatro corridas del notebook de Semana 11 en el MLflow del
devcontainer, desde `psets/proyecto/`:

```bash
python -m modelado.entrenar_modelos --tracking-uri http://127.0.0.1:5050
```

El script reutiliza por defecto la muestra de
`laboratorios/semana-11-sesion-02/starter/` si todavía no existe un CSV en
`data/processed/`. También acepta `--data` y `--tracking-uri`. Comprueba la
huella SHA-256 contra el manifiesto cuando está disponible, selecciona las
características solo con entrenamiento y registra los modelos
`linea_base_dummy`, `bosque_aleatorio`, `gradient_boosting` y
`red_neuronal_mlp` bajo el experimento
`Antes de que falle - SSD MC1 - TARGET_30D`. La elección final usa solo
PR-AUC y F0.5 de validación; la prueba se informa después de la elección.
Cada corrida contiene parámetros, métricas, procedencia y el artefacto
`pipeline_entrenado`. En `data/processed/modelo_final.json` quedan el `run_id`
final, las columnas y la huella de la muestra. Esos archivos locales no se
suben a Git.

Esta ejecución usó scikit-learn 1.5.2, la versión instalada en el
devcontainer del curso. Los notebooks adjuntos se ejecutaron con 1.7.2;
el servicio fija 1.5.2 para poder cargar el modelo recién entrenado.

## Modelo final

La corrida final de esta reproducción fue `red_neuronal_mlp`, `run_id`
`e9376de97c95409b9413647691551266`, con PR-AUC de validación 0.039071 y
PR-AUC de prueba 0.013095. Se registró como versión 1 de
`antes_de_que_falle_mc1_clasificador` en el MLflow del devcontainer.
La siguiente ejecución generará un nuevo `run_id`; el valor vigente queda en
`data/processed/modelo_final.json`.
La clasificación usa umbral fijo 0.50, igual que en los notebooks; las
probabilidades pueden requerir calibración antes de automatizar decisiones.

## Cómo correr el servicio

Desde `psets/proyecto/`, después de entrenar:

```bash
python -m servicio.exportar_modelo --tracking-uri http://127.0.0.1:5050
docker build -f servicio/Dockerfile -t mc1-servicio .
docker run --rm -p 8000:8000 mc1-servicio
```

`exportar_modelo.py` toma el `run_id` elegido en
`data/processed/modelo_final.json`, descarga el artefacto
`runs:/<run_id>/pipeline_entrenado` y genera `servicio/modelo.joblib`.
También acepta `--run-id` para exportar una corrida anterior. La imagen
incluye el modelo exportado y la misma versión de scikit-learn usada al
entrenar en el devcontainer. Para desarrollo directo:
`uvicorn servicio.api:app --port 8000`.

`GET /salud` indica si el artefacto está cargado e informa el `run_id`.
`POST /predecir` recibe `ssd_key`, `observation_date` y un objeto
`caracteristicas` con las ventanas SMART de 7, 14 y 30 días calculadas por la
OBT. Por ejemplo, el cuerpo de la petición se puede crear a partir de una fila
de `OBT_MC1_RN` omitiendo `TARGET_30D`, `PARTICION` y datos posteriores a la
fecha de observación. Las 54 columnas candidatas se documentan en
`modelado/pipeline_proyecto.py`; la API exige las columnas seleccionadas por el
modelo exportado y rechaza con 422 cualquier valor faltante de ellas. La
respuesta incluye la probabilidad de falla en 30 días, la decisión de revisión
y el `run_id` que permite rastrear la predicción. Los datos de entrenamiento,
los resultados y el registro MLflow no se copian a la imagen.

La API usa `preparar_observacion_inferencia` importada de
`modelado.pipeline_proyecto` para normalizar y ordenar las columnas. Las
ventanas SMART no pueden recalcularse a partir de una sola petición: la OBT
debe aportar el historial previo y las ventanas ya construidas.

Para reproducir una petición real con una fila completa de la partición de
prueba, en otra terminal del devcontainer:

```bash
cd psets/proyecto
python -m servicio.crear_peticion_ejemplo
curl -X POST http://127.0.0.1:8000/predecir \
  -H 'Content-Type: application/json' \
  --data-binary @data/processed/peticion_ejemplo.json
```

La petición generada contiene `ssd_key`, `observation_date` y las 41 ventanas
SMART seleccionadas, sin `TARGET_30D`. En esta reproducción se usó el puerto
18000 del devcontainer porque 8000 ya estaba ocupado; la respuesta fue:

```json
{"ssd_key":"000134b167e1ff263245170d1a1aeb0aa49e8f6a4265781229acfcb425dd2ee4","observation_date":"2019-11-05","probabilidad_falla_30d":0.046156090183027146,"requiere_revision":false,"run_id":"e9376de97c95409b9413647691551266"}
```
