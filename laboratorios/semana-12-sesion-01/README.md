# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

**`Lectura no disponible, enlace dañado.`**

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

**`Lectura no disponible, enlace dañado.`**

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

En este caso, ninguno de los dos debería de ser calculado por la noche. En el caso de predecir la tarifa de un viaje antes de pedirlo necesita información que solo se obtiene cuando un viaje se ha solicitado, lo que involucra que el proceso de predicción debe iniciar al mismo tiempo que una nueva solicitud de una carrera en base de los atributos temporales, geográficos, entre otros, disponibles al momento de la solicitud. Si se implementara un modelo de *batch prediction* para las tarifas de los taxis, las tarifas predichas no serían útiles para nadie, porque se prediciría con data previa, con viajes ya terminados, y a menos que exista un mecanismo de reajuste del cobro posterior al viaje, la predicción es inutil si el modelo trabaja en tiempo real. 

La idea de un modelo *batch prediction* si resulta útil si tratamos el problema de predecir en la noche no como un proceso hacia el cliente, pero un proceso de optimización adicional. Si el modelo predice, pero un sistema *onboard* como un taxímetro trabaja para determinar el costo real de la ruta en base del recorrido y velocidad del vehículo, entonces podríamos usar el sistema *onboard* como una *source of truth* y usar el estimado predicho por la noche como una guía para evaluar el error del modelo y utilizar los casos reales como fuente de optimización y reentrenamiento del modelo.

En el caso de la predicción de fraude, la mayoría de sistemas que implementan esta tecnología realizan la predicción segundos después de una compra, por lo que para las transacciones de pagos para un viaje se requeriría el mismo nivel de interactividad y rapidéz del modelo de predicción. En este caso, este modelo también tiene que ser en tiempo real, con los datos de una transacción realizada, y no podría usarse por la noche porque una transacción fraudolenta se espera ser marcada y notificada lo más rápido posible.

Ambos casos tienen que responder relativamente rápido. Dependiendo del modelo de aplicación de la predicción de la trafia es posible que este modelo tenga que responder lo más rápido posible si su predicción se usa como un estimado que se muestra al usuario, pero puede demorarse un poco si necesita información adicional sobre el trayecto para presentar un estimado. En el caso de la predicción de fraudes la información tiene que ser aún más rápida, como un sistema bancario la transacción debe ser notificada una vez se ha ejecutado para que se pueda tomar acciones correctivas inmediatamente.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

**`Realize la programación sin ejectuar esta prueba`**

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Esperaria que el modelo intente predecir una tarifa dependiendo del modelo implementado. Los modelos de árboles son capaces de aprender relaciones entre features hacia el objetivo inclusive cuando tienen valores nulos o faltantes. En el caso de nuestro proyecto, los modelos no tuvieron problema con valores nulos en las columnas de entrenamiento por lo que espero que una predicción se realize inclusive sin estas features disponibles.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Cada fila lleva la identificación del modelo porque esta se utiliza como un registro de quien realizó la predicción realizada, en lugar de tener metadata adicional que permita filtrar el modelo utilizado, cada fila mantiene esta información, utilizable para auditar el proceso de predicción realizado en auditorias, etc.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

1. En el caso de que se modifique el modelo y este no tolere los valores nulos, es posible que falle o no prediga en aquellos casos donde existen variables nulas, por lo que un cambio en el modelo podría eliminar la noción del TODO 4 que tenga la misma longitud de predicciones que ingresos, y que tenga la misma cantidad de IDs.
2. En el caso de que se modifique el modelo, el TODO 5 también corre el riesgo de tener nulos al final en las predicciones, o valores faltantes que no fueron predichos porque el modelo no tolera valores nulos.

**Lo que mostró el lote:**

```console
usr/bin/python3.11 /IdeaProjects/data-science-end-to-end/laboratorios/semana-12-sesion-01/starter/ejercicio1/predecir_lote.py 
Modelo 3154797e1ba6: 60 viajes leidos de viajes_programados.csv, 60 filas en tarifas_estimadas.csv
 id_viaje  tarifa_estimada version_modelo
    90001            27.42   3154797e1ba6
    90002            40.48   3154797e1ba6
    90003             6.37   3154797e1ba6
    90004             7.25   3154797e1ba6
    90005            40.34   3154797e1ba6

Process finished with exit code 0
```

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

```terminaloutput
/usr/bin/python3.11 /IdeaProjects/data-science-end-to-end/laboratorios/semana-12-sesion-01/starter/ejercicio2/comprobar.py 
Paso 1: OK
Paso 2: OK
Paso 3: OK
Paso 4: OK
....                                                                     [100%]
4 passed in 0.02s
Paso 5: OK

Process finished with exit code 0
```

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

```terminaloutput
/usr/bin/python3.11 /IdeaProjects/data-science-end-to-end/laboratorios/semana-12-sesion-01/starter/ejercicio2/puntuar_pagos.py 
Modelo f3038612d4f2: 300 pagos leidos de pagos_ayer.csv, 300 filas en pagos_puntuados.csv
Cola de revision (umbral 0.1): 26 pagos en cola_revision.csv
 id_pago  probabilidad_fraude version_modelo
   70058               0.9984   f3038612d4f2
   70222               0.9952   f3038612d4f2
   70063               0.9849   f3038612d4f2
   70170               0.9820   f3038612d4f2
   70059               0.9819   f3038612d4f2

Process finished with exit code 0
```

En este caso tenemos 26 pagos en la cola de revision cuya puntuacion de fraude corresponde a más del 10% del umbral.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

La empresa pierde la capacidad de alertar rápidamente a un cliente de un cobro que parece ser sospechoso y fraudolento, si la empresa mantiene esta información y solo la revisa en la mañana luego de un incidente, esta información no ayuda a los clientes, y por tanto puede resultar en alarmas tardías para clientes y la empresa.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

La prueba gana una ejecución completamente diferente de los valores de entrenamiento y prueba ya usados. La prueba permite evidenciar el comportamiento de la cola de revisión, sin utilizar información que ya fue usada dentro de la prueba y pueda resultar en bias hacia los datos de prueba. 


## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

[_(tu respuesta)_](https://github.com/Volv1t3/data-science-end-to-end/actions/runs/42870400918)


## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

_(tu respuesta)_
