# Semana 13, Sesión 2: Monitoreo en producción I: métricas de rendimiento

> Objetivo de la sesión: definir las métricas de monitoreo del modelo de tu proyecto en producción, distinguiendo métricas de sistema (latencia, disponibilidad) de métricas de modelo (desempeño).

**Lectura previa de esta sesión:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 8: Data Distribution Shifts and Monitoring.


## Ejercicio 1: la tarifa, instrumentada

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa con la versión 1 del modelo (la red de 64 neuronas), ahora con dos agregados. Un middleware (una función que corre alrededor de cada petición) mide cuánto tarda cada predicción y con qué código responde; `/metricas` resume esas mediciones. Y cada predicción queda guardada en `predicciones.jsonl`, con un identificador que la app guarda junto al viaje. [`simular_noviembre.py`](./starter/ejercicio1/simular_noviembre.py) manda al servicio los 1156 viajes de noviembre y escribe `cobros.csv`, lo que el sistema de cobros anotó al terminar cada viaje: el identificador de la predicción y la tarifa cobrada (no lo modifiques). [`desempeno.py`](./starter/ejercicio1/desempeno.py) une las dos cosas y calcula el error por semana. Los 4 `TODO`: medir la latencia, resumir las métricas, guardar cada predicción y calcular el error por semana.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python simular_noviembre.py
python desempeno.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** los 1156 viajes recibieron su tarifa. ¿Qué dicen `/metricas` y `desempeno.py`? ¿Qué podrías afirmar sobre el servicio en noviembre?

En este caso, la implementacion de `simular_noviembre.py` usado para calcular el valor de la tarifa de prueba de noviembre utiliza su propia version de la API alzada directamente desde el sistema de generacion de las tarifas, y no interactua con el sistema desplegado manualmente a traves de uvicorn, por lo tanto, el sistema, aunque existe un apartado de 
metricas, no obtiene informacion desde la API continua, sino que utiliza informacion delimitada y efimera solo para este proceso. 

```terminaloutput
(.venv) vscode ➜ .../laboratorios/semana-13-sesion-02/starter/ejercicio1 (laboratorio/semana-13-02) $ python3 simular_noviembre.py 
1156 viajes simulados, del 2026-11-02 al 2026-11-28
Metricas del sistema: {'volumen_peticiones': 1159, 'errores_validacion_datos': 3, 'errores_servidor': 0, 'latencia_p95_ms': 3.19, 'latencia_p99_ms': 3.66}
```

 Como puede verse en el resultado de las metricas del sistema efimero, se registraron 1159 viajes de los cuales solo 1156 tuvieron una clasificacion, indicando 3 errores de validacion. Las solicitudes se ejecutaron con una latencia p95 de 3.19 ms por lo que el 95 de las solicitudes tuvieron un tiempo de respuesta menor a 3.19 ms en promedio. 

Estos resultados nos indican que la implementacion del servicio es apropiada, y que las mtricas del sistema se registran correctamente. Tambien tome nota del resultado antes de hacer el TODO 1 que registra 
las metricas, el resultado fue


```terminaloutput
(.venv) vscode ➜ .../laboratorios/semana-13-sesion-02/starter/ejercicio1 (laboratorio/semana-13-02) $ python3 simular_noviembre.py 
1156 viajes simulados, del 2026-11-02 al 2026-11-28
Metricas del sistema: {}
```
Durante noviembre, el servicio demostro su capacidad de responder a las solicitudes de predicciones de clientes con una latencia menor a los 3.19 ms en el 95 de las solicitudes. El sistema demostro su capacidad de defenderse frente a solcitudes con formatos incorrectos que pudieran causar problemas en la prediccion a traves de respuestas con errores de validacion.

**Respuesta 2 (TODO 1 y 2):** antes de correr la simulación, ¿qué latencia p95 esperas, y cuántos errores de validación? ¿Esperas que la latencia cambie entre la primera y la última semana del mes?

La latencia del modelo deberia ser muy baja, en los millisegundos  en general por el hecho de cargar un modelo liviano, y convertir un objeto pequeno de datos en las matrices necesarias para la inferencia. En este caso, no puedo dar con seguridad una cantidad de errores de validacion, el codigo indica que se generan tres solicitudes erroneas directamente, lo cual resultaria en 3 solicitudes incorrectas generadas, pero no puedo saber directamente cuantos errores habra. 

Debido a que el modelo tiene que cargarse una vez durante inicializacion de la API, la latencia deberia reducirse a solamente la serializacion y deserializacion HTTP y la transformacion de los datos a los esperados por el modelo, por lo que la latencia deberia estandarizarse en un rango especifico cuando el modelo ya esta precargado en memoria.


**Respuesta 3 (TODO 3):** el servicio devuelve `id_prediccion` y lo guarda en el registro. ¿Para qué lo necesita el sistema de cobros? ¿Qué pasaría si solo se guardaran la fecha y la tarifa estimada?

El ID de clasificacion pertmie implementar un sistema de trazabilidad entre las aplicaciones de la empresa o clientes que dependan de las clasificaciones, y las clasificaciones registradas en el sistema. La aplicacion permite realizar una auditoria de los resultados sea internamente o por agentes externos que se perderian si solo se guardara la fecha y la tarifa estimada.

**Respuesta 4 (TODO 4):** el MAE en prueba de la versión 1 fue 1.17. Antes de correr `desempeno.py`, ¿qué esperas para cada semana de noviembre?

```terminaloutput
(.venv) vscode ➜ .../laboratorios/semana-13-sesion-02/starter/ejercicio1 (laboratorio/semana-13-02) $ python3 entrenar.py 
Version 1 guardada en modelo_tarifa.joblib, MAE en prueba 1.1695
```

Este fue el resultado del entrenamiento, para corroborar los datos de la pregunta. 

Teniendo en cuenta la naturaleza degenerativa de los modelos en produccion al estar expuestos a nuevas distribuciones de datos, y a distribuciones de estos nunca antes vistas, esperaria que el MAE se mantenga relativamente cerca del calculado durante la prueba del modelo, pero con una ligera tendencia a decrecer entre registros diarios. 

```terminaloutput
(.venv) vscode ➜ .../laboratorios/semana-13-sesion-02/starter/ejercicio1 (laboratorio/semana-13-02) $ python3 desempeno.py
===============================================Resultado Luego del Merge================================================
                         id_prediccion  tarifa      prediccion_created_at  version_modelo    fecha_hora_inicio      zona_origen     zona_destino distancia_km pasajeros metodo_pago  tarifa_cobrada
0     49a5ec6dde4f44a9a98f95abb4a155ed   14.86 2026-11-20 15:35:40.637269               1  2026-11-02 00:14:00          Midtown  Upper East Side         4.19      None     tarjeta           14.36
1     b218541035394a8388dbfc2973d0faa9   70.27 2026-11-20 15:35:40.641697               1  2026-11-02 00:14:00      JFK Airport          Midtown        23.49       1.0     tarjeta           70.00
2     e4ead672fd304e78950bc956e9ab6458   70.51 2026-11-20 15:35:40.645641               1  2026-11-02 02:21:00  Upper East Side      JFK Airport        31.02       2.0    efectivo           70.00
3     5edb9155d1be4f8b8cad1f67c46530cf   16.11 2026-11-20 15:35:40.649393               1  2026-11-02 02:47:00  Upper East Side          Midtown         4.54       4.0     tarjeta           14.08
4     24f4da6b11b84f2fad21720d8bd61fa5   28.31 2026-11-20 15:35:40.653072               1  2026-11-02 03:36:00  Upper East Side         Brooklyn         9.19       4.0     tarjeta           25.43
...                                ...     ...                        ...             ...                  ...              ...              ...          ...       ...         ...             ...
1151  1e1e4e62e2e046a6b4b17718e00b6641    6.53 2026-11-20 15:35:44.596085               1  2026-11-28 22:11:00  Upper East Side  Upper East Side         0.71       3.0    efectivo            5.50
1152  d8715dd827c54157b09af04f87418f4e   22.09 2026-11-20 15:35:44.599392               1  2026-11-28 22:19:00          Midtown           Harlem         6.98       2.0     tarjeta           26.51
1153  c3445d59c66d45dc83448a039ae76e62   20.25 2026-11-20 15:35:44.603062               1  2026-11-28 22:25:00          Midtown           Queens         6.12       1.0     tarjeta           23.68
1154  74475a76bfef42119200a00f76b1d3b7   11.55 2026-11-20 15:35:44.606512               1  2026-11-28 22:58:00  Upper West Side  Upper West Side         2.65       2.0     tarjeta            9.34
1155  ed165f6982a94f0da30ae39b4915ca2d   24.38 2026-11-20 15:35:44.609866               1  2026-11-28 23:04:00  Lower Manhattan  Upper East Side         8.17       1.0     tarjeta           26.66

[1156 rows x 11 columns]
===============================================Resultado Luego del Merge================================================

===========================================Resultado Luego del Error Absoluto===========================================
                         id_prediccion  tarifa      prediccion_created_at  version_modelo    fecha_hora_inicio      zona_origen     zona_destino distancia_km pasajeros metodo_pago  tarifa_cobrada  error_absoluto
0     49a5ec6dde4f44a9a98f95abb4a155ed   14.86 2026-11-20 15:35:40.637269               1  2026-11-02 00:14:00          Midtown  Upper East Side         4.19      None     tarjeta           14.36            0.50
1     b218541035394a8388dbfc2973d0faa9   70.27 2026-11-20 15:35:40.641697               1  2026-11-02 00:14:00      JFK Airport          Midtown        23.49       1.0     tarjeta           70.00            0.27
2     e4ead672fd304e78950bc956e9ab6458   70.51 2026-11-20 15:35:40.645641               1  2026-11-02 02:21:00  Upper East Side      JFK Airport        31.02       2.0    efectivo           70.00            0.51
3     5edb9155d1be4f8b8cad1f67c46530cf   16.11 2026-11-20 15:35:40.649393               1  2026-11-02 02:47:00  Upper East Side          Midtown         4.54       4.0     tarjeta           14.08            2.03
4     24f4da6b11b84f2fad21720d8bd61fa5   28.31 2026-11-20 15:35:40.653072               1  2026-11-02 03:36:00  Upper East Side         Brooklyn         9.19       4.0     tarjeta           25.43            2.88
...                                ...     ...                        ...             ...                  ...              ...              ...          ...       ...         ...             ...             ...
1151  1e1e4e62e2e046a6b4b17718e00b6641    6.53 2026-11-20 15:35:44.596085               1  2026-11-28 22:11:00  Upper East Side  Upper East Side         0.71       3.0    efectivo            5.50            1.03
1152  d8715dd827c54157b09af04f87418f4e   22.09 2026-11-20 15:35:44.599392               1  2026-11-28 22:19:00          Midtown           Harlem         6.98       2.0     tarjeta           26.51            4.42
1153  c3445d59c66d45dc83448a039ae76e62   20.25 2026-11-20 15:35:44.603062               1  2026-11-28 22:25:00          Midtown           Queens         6.12       1.0     tarjeta           23.68            3.43
1154  74475a76bfef42119200a00f76b1d3b7   11.55 2026-11-20 15:35:44.606512               1  2026-11-28 22:58:00  Upper West Side  Upper West Side         2.65       2.0     tarjeta            9.34            2.21
1155  ed165f6982a94f0da30ae39b4915ca2d   24.38 2026-11-20 15:35:44.609866               1  2026-11-28 23:04:00  Lower Manhattan  Upper East Side         8.17       1.0     tarjeta           26.66            2.28

[1156 rows x 12 columns]
===========================================Resultado Luego del Error Absoluto===========================================

==========================================Resultado Luego de Anadir la Semana===========================================
                         id_prediccion  tarifa      prediccion_created_at  version_modelo   fecha_hora_inicio      zona_origen     zona_destino distancia_km pasajeros metodo_pago  tarifa_cobrada  error_absoluto      semana
0     49a5ec6dde4f44a9a98f95abb4a155ed   14.86 2026-11-20 15:35:40.637269               1 2026-11-02 00:14:00          Midtown  Upper East Side         4.19      None     tarjeta           14.36            0.50  2026-11-02
1     b218541035394a8388dbfc2973d0faa9   70.27 2026-11-20 15:35:40.641697               1 2026-11-02 00:14:00      JFK Airport          Midtown        23.49       1.0     tarjeta           70.00            0.27  2026-11-02
2     e4ead672fd304e78950bc956e9ab6458   70.51 2026-11-20 15:35:40.645641               1 2026-11-02 02:21:00  Upper East Side      JFK Airport        31.02       2.0    efectivo           70.00            0.51  2026-11-02
3     5edb9155d1be4f8b8cad1f67c46530cf   16.11 2026-11-20 15:35:40.649393               1 2026-11-02 02:47:00  Upper East Side          Midtown         4.54       4.0     tarjeta           14.08            2.03  2026-11-02
4     24f4da6b11b84f2fad21720d8bd61fa5   28.31 2026-11-20 15:35:40.653072               1 2026-11-02 03:36:00  Upper East Side         Brooklyn         9.19       4.0     tarjeta           25.43            2.88  2026-11-02
...                                ...     ...                        ...             ...                 ...              ...              ...          ...       ...         ...             ...             ...         ...
1151  1e1e4e62e2e046a6b4b17718e00b6641    6.53 2026-11-20 15:35:44.596085               1 2026-11-28 22:11:00  Upper East Side  Upper East Side         0.71       3.0    efectivo            5.50            1.03  2026-11-23
1152  d8715dd827c54157b09af04f87418f4e   22.09 2026-11-20 15:35:44.599392               1 2026-11-28 22:19:00          Midtown           Harlem         6.98       2.0     tarjeta           26.51            4.42  2026-11-23
1153  c3445d59c66d45dc83448a039ae76e62   20.25 2026-11-20 15:35:44.603062               1 2026-11-28 22:25:00          Midtown           Queens         6.12       1.0     tarjeta           23.68            3.43  2026-11-23
1154  74475a76bfef42119200a00f76b1d3b7   11.55 2026-11-20 15:35:44.606512               1 2026-11-28 22:58:00  Upper West Side  Upper West Side         2.65       2.0     tarjeta            9.34            2.21  2026-11-23
1155  ed165f6982a94f0da30ae39b4915ca2d   24.38 2026-11-20 15:35:44.609866               1 2026-11-28 23:04:00  Lower Manhattan  Upper East Side         8.17       1.0     tarjeta           26.66            2.28  2026-11-23

[1156 rows x 13 columns]
==========================================Resultado Luego de Anadir la Semana===========================================
       semana  viajes       mae  veces_la_referencia
0  2026-11-02     291  1.179244                 1.01
1  2026-11-09     318  1.340346                 1.15
2  2026-11-16     308  4.291591                 3.67
3  2026-11-23     239  4.304226                 3.68
```


## Ejercicio 2: el fraude, cuando la verdad tarda tres semanas

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de fraude guardó los 5874 pagos que evaluó en noviembre en [`predicciones_fraude.csv`](./starter/ejercicio2/predicciones_fraude.csv): la probabilidad, si mandó el pago a revisión y cuánto tardó. Un fraude se confirma cuando el titular de la tarjeta reclama el cobro, entre 2 y 21 días después; los confirmados están en [`contracargos.csv`](./starter/ejercicio2/contracargos.csv). Hoy es 14 de diciembre. [`monitoreo_fraude.py`](./starter/ejercicio2/monitoreo_fraude.py) arma el reporte del mes. Los pasos 1 a 3 tienen `TODO`: la latencia p95 por día, la tasa de alertas por semana y los fraudes confirmados hasta hoy. Los pasos 4 (el recall de las semanas cuyos fraudes ya se confirmaron) y 5 (las alertas) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python monitoreo_fraude.py`.

**Respuesta 1:** el reporte da tres tipos de alerta. ¿Cuáles son de sistema y cuáles de modelo? ¿El día de la latencia alta coincide con alguna semana de recall bajo?

_(tu respuesta)_

**Respuesta 2:** la tasa de alertas se puede calcular el mismo día; el recall, tres semanas después. ¿Qué semana mostró primero el problema la tasa de alertas, y cuándo lo confirmó el recall? ¿Por qué la tasa de alertas no alcanza para decir que el modelo empeoró?

_(tu respuesta)_

**Respuesta 3:** ¿por qué el reporte no muestra el recall de la semana del 23 de noviembre, si ya hay fraudes confirmados de esa semana?

_(tu respuesta)_

**Si terminas antes:** cambia `HOY` a `"2026-11-30"` y vuelve a correr. ¿Qué semanas tienen recall? ¿Desde qué día el reporte puede mostrar la caída del recall de la semana del 16?

_(tu respuesta)_

## Ejercicio 3: el monitoreo de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio de las dos sesiones anteriores. Esta vez no hay plantilla.

### Entregable 1: el servicio, instrumentado

- Un middleware que mida la latencia y el código de cada predicción, y una ruta `/metricas` con, al menos, la cantidad de peticiones, los errores de validación, los errores de servidor y la latencia p50 y p95.
- Cada predicción queda en un registro (un archivo JSONL en `monitoreo/`, por ejemplo) con su identificador, el momento, la versión del modelo, la entrada y la salida. Lo que haga falta para unirla después con la verdad tiene que estar en ese registro.
- Pruebas en `servicio/test_api.py` que comprueben que una predicción queda registrada y que `/metricas` separa los errores de validación de los de servidor, con un registro temporal (`tmp_path`).

### Entregable 2: `monitoreo/desempeno.py`

Une el registro con la verdad de tu proyecto y calcula la métrica de la Semana 11 por período (día o semana) y por versión. Si en tu proyecto la verdad no llega, o llega muy tarde, define una métrica que se pueda calcular antes y explica qué te dice y qué no.

### Entregable 3: `docs/monitoreo.md`

Una tabla con cada métrica que vigilas: si es de sistema o de modelo, de dónde sale, cuándo se conoce y con qué umbral alerta.

**¿Cuánto tarda en llegar la verdad en tu proyecto, y qué métrica te avisa mientras tanto?**

_(tu respuesta)_

**¿Qué falla de tu modelo no vería ninguna de tus métricas de sistema?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe el servicio son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

_(tu respuesta)_
