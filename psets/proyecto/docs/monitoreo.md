# Monitoreo del clasificador SSD MC1

El servicio estima la probabilidad de falla de un SSD durante los 30 días
posteriores a `observation_date`. La decisión de revisión usa el umbral 0.50.
Se vigilan por separado la disponibilidad de la API y el desempeño del
modelo. Una API rápida puede seguir clasificando mal.

## Métricas y alertas

| Métrica | Tipo | Fuente | Cuándo se conoce | Umbral de alerta |
|---|---|---|---|---|
| Disponibilidad de `/salud` | Sistema | Sondeo externo del contenedor | Cada sondeo | Dos sondeos consecutivos fallidos; el sondeo externo debe configurarse en el entorno de despliegue. |
| Peticiones de predicción | Sistema | Middleware de `/predecir`, ruta `/metricas` | Al responder | Cero durante un lote diario programado; contrastar con el inventario esperado. |
| Errores de validación (422) | Sistema / calidad de entrada | Middleware, `/metricas` | Al responder | Más del 1 % de las peticiones del lote. |
| Errores de servidor (5xx) | Sistema | Middleware, `/metricas` | Al responder | Cualquier error 5xx. |
| Latencia p50 y p95 | Sistema | Middleware, `/metricas` | Al responder | p50 > 100 ms o p95 > 200 ms durante un lote. Límites iniciales, sujetos a medir en el entorno real. |
| Tasa de alertas (`requiere_revision`) | Modelo, señal indirecta | JSONL de predicciones; `monitoreo/desempeno.py` | Inmediatamente | Con al menos 100 predicciones semanales por versión, fuera de 1,7 %–6,9 %. El intervalo provisional es la mitad y el doble del 3,47 % observado en validación para la versión 1. |
| PR-AUC (average precision) por semana y versión | Modelo, desempeño con verdad | JSONL unido con `SSD_KEY`, `OBSERVATION_DATE` y `TARGET_30D` de la OBT finalizada | Al menos 30 días después de la observación, más la demora de ingesta | Con al menos 200 casos etiquetados y 5 fallas confirmadas, < 0,0293 (75 % de la PR-AUC 0,039071 de validación de la versión 1). Investigar antes de promover o revertir. |
| Cobertura de verdad madura | Calidad del monitoreo | Unión entre JSONL y OBT finalizada | Después de 30 días más ingesta | < 95 % de los casos que ya completaron el horizonte. |

Los umbrales son reglas de investigación iniciales, no garantías estadísticas.
La muestra de entrenamiento está sobremuestreada y las probabilidades pueden
estar mal calibradas. Una tasa de alertas estable tampoco garantiza PR-AUC
estable: puede cambiar la relación entre señales SMART y fallas sin que
cambien las distribuciones de entrada o salida.

## Registro y cálculo

Cada predicción exitosa recibe `id_prediccion` y añade una línea a
`monitoreo/predicciones.jsonl`: momento UTC, versión, `run_id`, entrada
(incluidos `ssd_key`, `observation_date` y ventanas SMART) y salida. El
identificador permite auditar una decisión; la pareja SSD-fecha permite
unirla con la etiqueta posterior. Una predicción cuyo registro no pueda
escribirse falla, para no entregar una respuesta sin trazabilidad.

`/metricas` cuenta solo llamadas a `/predecir`. Sus contadores son
acumulados por proceso; los percentiles usan las últimas 10 000 llamadas.
Se reinician al reiniciar el proceso y no agregan varias réplicas. Para
operación continua hace falta exportarlos a un recolector externo. El JSONL
debe conservarse en un volumen y protegerse como telemetría operativa.

Desde `psets/proyecto/`, para levantar la imagen con un registro persistente:

```bash
python -m servicio.exportar_modelo
docker build -f servicio/Dockerfile -t mc1-servicio .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/monitoreo:/app/monitoreo" mc1-servicio
```

El reporte puede ejecutarse antes de recibir etiquetas; entonces muestra
volumen, tasa de alertas y probabilidad media, y deja PR-AUC vacía:

```bash
python -m monitoreo.desempeno
```

Cuando la OBT tenga etiquetas finalizadas, se pasa un CSV con `SSD_KEY`,
`OBSERVATION_DATE` y `TARGET_30D`:

```bash
python -m monitoreo.desempeno --verdad data/processed/verdad_mc1.csv
```

Para comprobar el mecanismo con la muestra histórica de 2019:

```bash
python -m monitoreo.desempeno \
  --verdad ../../laboratorios/semana-11-sesion-02/starter/mc1_muestra_experimentos.csv \
  --fecha-corte 2019-12-31
```

El script agrupa por semana de observación y `version_modelo`, guarda
`data/processed/monitoreo_semanal.csv` y calcula PR-AUC solo si el horizonte
de 30 días terminó y existen ambas clases. `--periodo dia` permite cambiar
la agrupación. Conserva todos los reintentos en el JSONL, pero cuenta solo
la última predicción de cada SSD-fecha-versión en las métricas del modelo.
La muestra histórica sirve para probar la unión; no representa
una medición de producción de 2026.
