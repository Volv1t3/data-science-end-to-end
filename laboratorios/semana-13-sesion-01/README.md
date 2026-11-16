# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

En este caso, si la version dice NONE desde el archivo de listado de versiones y promover resulta en un error, esto significa que al mometno de ejecutar estos archivos, no existe ningun tipo de informacion en el registro local de MLFLOW que podria intentar usar la aplicacion para cargar un modelo para el servicio. Como la versiones son None entonces no hay ningun modelo registrado, y como `promover.py` requiere 
tener una lista viva, tambien da error.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

En clase registramos tres versiones con 100 neuronas, adicionalmente registro la de 64 y la de 8. En totalidad se esperaria cinco versiones, habria tres base de la clase y dos 
adicionales de esta instruccion.

Debido a que existian diferentes registros ya bajo el mismo modelo, la version fue modificada por el sistema, registrando el modelo de 64 nueronas con la version 4, y aquel de 8 con la version 5. Este comportamiento tiene sentido debido a que tenemos el mismo nombre del modelo registrado bajo el mismo experimento. 

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

En este caso,  si alzamos en produccion con la version 2, tenemos que pasar por listar y luego por promover.

```console
/IdeaProjects/data-science-end-to-end/.venv/bin/python /IdeaProjects/data-science-end-to-end/laboratorios/semana-13-sesion-01/starter/ejercicio1/versiones.py 
Version 1: red de 100 neuronas, MAE 1.1950, alias: -
Version 2: red de 100 neuronas, MAE 1.1950, alias: -
Version 3: red de 100 neuronas, MAE 1.1950, alias: -
Version 4: red de 64 neuronas, MAE 1.1695, alias: -
Version 5: red de 8 neuronas, MAE 1.1877, alias: -

Process finished with exit code 0
```

Ahora en base  a esta informacion vamos a hacer una promocion a produccion del 
modelo de version 2 (mi version 5) que corresponde al de 8 neuronas.

```console
/IdeaProjects/data-science-end-to-end/.venv/bin/python /IdeaProjects/data-science-end-to-end/laboratorios/semana-13-sesion-01/starter/ejercicio1/promover.py 5 
tarifa-taxis@produccion es ahora la version 5 (corrida f351c193b731)

Process finished with exit code 0
```

Con esto registrado, ahora vamos a definir la request con el servicio en ejecucion

```console 
{
  "estado": "ok",
  "modelo_cargado": true,
  "version_modelo": "5"
}

```

La razon por la que el sistema marca que el modelo cargado es el version 5 es porque durante la clase se realizaron mas ejecuciones y por tanto la version 2 de este documento es la version 5 del sistema al ser el ultimo registro ejecutado.


**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

En este caso, antes del TODO 4, la aplicacion iba a fallar desde el intenot de realizar la preuba del endpoint de salud. La aplicacion se iniciaba sin un modelo por lo que el 
assert del codigo HTTP de la respuesta podria pasar, pero el assert del cuerpo de la respuesta no podria pasar nunca debido a que el modelo no esta cargado 
y por tanto la respuesta no seria igual al cuerpo esperado.

Las pruebas validan que el modelo cargado tenga como version 7, mientras que 
el endpoint de salud no verifica la version cargada, las pruebas definidas para la aplicacion son mas exigentes en las revisiones que  la propia revision en produccion, y por tanto comprueban la version del modelo cargado de manera mas exigente

**Lo que respondió el servicio:**

Para el modelo de la version 5 (la version dos de este laboratorio)

```
{
  "fecha_hora_inicio": "2026-11-16T20:38:15.036Z",
  "zona_origen": "Midtown",
  "zona_destino": "JFK Airport",
  "distancia_km": 20,
  "pasajeros": 2,
  "metodo_pago": "tarjeta"
}

-> 
{
  "tarifa_estimada": 69.97,
  "version_modelo": "5"
}
```

Para el modelo de la version 4 (1 de esta clase)

```
{
  "tarifa_estimada": 70.13,
  "version_modelo": "4"
}
```

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

```
/usr/bin/python3 /IdeaProjects/data-science-end-to-end/laboratorios/semana-13-sesion-01/starter/ejercicio2/ciclo_fraude.py 
Registered model 'fraude-pagos' already exists. Creating a new version of this model...
Created version '13' of model 'fraude-pagos'.
logistica: version 13, recall 0.644, precision 0.304, promovida: False
Registered model 'fraude-pagos' already exists. Creating a new version of this model...
Created version '14' of model 'fraude-pagos'.
gradient-boosting: version 14, recall 0.661, precision 0.429, promovida: False
Registered model 'fraude-pagos' already exists. Creating a new version of this model...
Created version '15' of model 'fraude-pagos'.
logistica-muy-regularizada: version 15, recall 0.102, precision 0.667, promovida: False
En produccion: version 11

Process finished with exit code 0

```

La version que qeudo en produccion fue la version 11, debido a que ejecute varias veces el check en ciclos. En este caso, la version 15 tiene un recall 0.102, y en este caso el recall que tenemos debe ser mas bajo que el de la version 11 que tiene un recall de 0.6610169491525424. como ninguna de las metricas resultates son superiores en general, esta ejecucion se mantiene en el sistema.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

Se rechaza este caso porque no estamos optimizando para los casos en donde el modelo acierta con respecto del total de casos positivos y falsos positivos, estamos tratando e optimizar los casos positivos en el total de los casos positivos del sistema, es decir, tanto casos correctamente clasificados y los falsos negativos sdel sistema. 

Para no rechazar este caso y aceptar esta entrada, se tendria que modificar la comparativa para utilizar tanto recall como precision y elegir la metrica prioritaria, o utilizar solamente la metrica de precision.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

La respuesta esta en las tags de comparativa, el resultado que se guarda en validacion, y comparada_con quien registra el ID de la entrada con la que fue comparada con. Esto permite a un evaluador revisar la trazabilidad completa de una decision de modificacion del modelo de produccion, y determinar cual fue el modelo que se mantuvo en produccion y quien fue rechazado.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

```
/usr/bin/python3 /IdeaProjects/data-science-end-to-end/laboratorios/semana-13-sesion-01/starter/ejercicio2/ciclo_fraude.py 
Registered model 'fraude-pagos' already exists. Creating a new version of this model...
Created version '13' of model 'fraude-pagos'.
logistica: version 13, recall 0.644, precision 0.304, promovida: False
Registered model 'fraude-pagos' already exists. Creating a new version of this model...
Created version '14' of model 'fraude-pagos'.
gradient-boosting: version 14, recall 0.661, precision 0.429, promovida: False
Registered model 'fraude-pagos' already exists. Creating a new version of this model...
Created version '15' of model 'fraude-pagos'.
logistica-muy-regularizada: version 15, recall 0.102, precision 0.667, promovida: False
En produccion: version 11

Process finished with exit code 0

```

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

_(tu respuesta)_

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

_(tu respuesta)_
