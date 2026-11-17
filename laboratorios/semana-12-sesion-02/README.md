# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).


## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

De esta salida deberia de preocuparme la existencia de una salida de este estilo. Al estar trabajnado con un sistema con un modelo cargado que no sabemos cual es, el principal problema es que diga modelo cargad ypase las pruebas sin saber que modelo se cargo, y de donde vino.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Si se registrara un caso de que viniera una solicitud sin `distancia_km` el resultado eneral seria direcaamente un errr interno de validacion. La respuesta no la enviaria directamente la app porque seria un error de validacion de pydantic.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

Si el modelo calculara las horas sin utilizar el mecanismo de la implementacino de entrenamiento, los datos que recibiria el modeloseai co,pleamente distnitos en su distribucion y posibles valores a los que aprendio durante el entrenamiento. Ademas, si las formulas no coinciden y el procesamiento varia, podria haber un drift de las predicciones del modelo resultando de la discrepancia entre el proeso de entrenameinto y despliegue.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

La peticion llega, pero el modelo que se registra lo intercepta primero Pydantic para la validacion del estado de los datos y si mantiene la estructura reqrida. EL error 422 Unprocessable Entity se usa desde Pydantic para indicar que existen errores en la definicion del objeto enviado al endpoint de la API que no permite que sea procesado.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque la exposicion del puerto con `-p 8000:8000` funciona, pero Uvicorn por defecto escucha solamente en `127.0.0.1` dentro del contenedor. Docker redirige las peticiones hacia la interfaz de red del contenedor y no hacia su loopback, por lo que no puede llegar al proceso aunque el puerto este mapeado. Al agregar `--host 0.0.0.0` al `CMD`, Uvicorn escucha en todas las interfaces del contenedor y Docker puede reenviar las peticiones de `localhost:8000` correctamente.


## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

El pago riesgoso recibio `{"probabilidad_fraude": 0.9996, "revisar": true}`. El pago tranquilo recibio `{"probabilidad_fraude": 0.0159, "revisar": false}`.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app puede usar `revisar` directamente para decidir si pausa el cobro, sin duplicar la regla del umbral. Si el umbral cambia a 0.20, se cambia la constante `UMBRAL` del servicio y se vuelve a desplegar; la probabilidad del modelo no cambia.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Debe dar la misma probabilidad si ambos usan el mismo modelo, las mismas columnas en el mismo orden y las mismas features derivadas. En particular, los dos deben calcular `hora` y `monto_por_km` con `agregar_features` antes de llamar al modelo.

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde 422 Unprocessable Entity con `float_parsing` en `body.monto`: no puede convertir `"cuarenta"` a numero. La validacion la produce la anotacion `monto: float` de la clase `Pago`; FastAPI y Pydantic rechazan la peticion antes de ejecutar `evaluar_pago`.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

El servicio recibe `ssd_key`, `observation_date` y las ventanas SMART de 7,
14 y 30 días ya calculadas por la OBT. La API utiliza
`preparar_observacion_inferencia` de `modelado.pipeline_proyecto` para validar,
convertir a números y ordenar las características seleccionadas por el modelo.
La OBT calcula las ventanas a partir del historial; una sola petición no
contiene ese historial. Entrenamiento e inferencia comparten los nombres y el
orden de columnas mediante el pipeline exportado.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

Desde `psets/proyecto/`, con el contenedor levantado en el puerto 18000 del
devcontainer (el 8000 estaba ocupado):

```bash
python -m servicio.crear_peticion_ejemplo
curl -X POST http://127.0.0.1:18000/predecir \
  -H 'Content-Type: application/json' \
  --data-binary @data/processed/peticion_ejemplo.json
```

La petición generada usa una observación completa de prueba, con el SSD
`000134b167e1ff263245170d1a1aeb0aa49e8f6a4265781229acfcb425dd2ee4`,
fecha `2019-11-05` y 41 ventanas SMART. Respuesta real:

```json
{"ssd_key":"000134b167e1ff263245170d1a1aeb0aa49e8f6a4265781229acfcb425dd2ee4","observation_date":"2019-11-05","probabilidad_falla_30d":0.046156090183027146,"requiere_revision":false,"run_id":"e9376de97c95409b9413647691551266"}
```

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

_(tu respuesta)_
