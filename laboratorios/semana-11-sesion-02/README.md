# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Teniendo en cuenta la implementación de la pdivisión única, esperaria que mediante la aplicación de *K-Fold Cross Validation* los errores de los cinco folds entrados deberían de ser similares entre sí. Al implementar esre método, forzamos al modelo a validarse con diferentes segmentos de los datos de entrenamiento, dejando un subconjunto de datos libres, la idea principal es reducir la posibilidad de que el modelo tenga un sobreajuste sobre los datos de entrenamiento. Al realizar diferentes combinaciones de validaciones sobre el split de los datos, y sobre los splits internos entre validación y entrenamiento, se esperaría una diferencia no elevada entre los diferentes pliegues ejecutados. 

Esperaría que varíen en un rango de 0.25 a 0.50 por Fold.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

En este caso, el error del split temporal  debería de aumentar debido a que la división que utiliza el método es incompatible con la naturaleza del problema de predicción. Al utilizar la KFold CV, los datos de observaciones tanto futuras como pasadas pueden utilizarse dentro del esquema de validación o de entrenamiento, debido a que el único limitante es que un dato sea parte de la valdiación una sola vez. 

Además, dado que KFold CV no establece una separación en rangos de antes y después de, la data tomada por este método sigue de manera cercana la distribución de lso datos, mientras que el método temporal intenta separar los datos en rangos de tiempo, lo que puede generar que el modelo no tenga la misma distribución de datos para entrenar y validar, generando un error mayor.


**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

Debido a que los datos de las tarifas registradas pueden tener a las zonas de origen más de una vez en sus registros, el peor error corresponde a este tipo de entrenamiento dado que si el mecanismo filtra los casos de entrada de un grupo o el otro, el momento en que tengamos que entrenar con casos de Queens, no tendremos nada con que validar esos mismos casos y eliminamos un grupo de datos de la validación totalmente, lo mismo sucede si ese registro esta solo en la validación, reduciendo los datos que tiene el modelo para utilizar por cada iteración.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes subestmados.

**Lo que confirmó la tabla, esquema por esquema:**

```terminaloutput
                  validacion  mae_promedio  desviacion                       pliegues
aleatorio, gradient boosting        1.2288      0.0361 [1.28, 1.24, 1.16, 1.24, 1.23]
           aleatorio, bosque        1.2887      0.0286  [1.3, 1.27, 1.25, 1.33, 1.29]
 temporal, gradient boosting        1.3099      0.0693  [1.4, 1.21, 1.38, 1.27, 1.29]
 por zona, gradient boosting        2.2996      2.0648 [1.27, 1.31, 1.29, 1.19, 6.43]

```

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

```terminaloutput
linea base           accuracy 0.9508   recall 0.0000   precision 0.0000
```

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

```terminaloutput
gradient boosting    accuracy 0.9712   recall 0.4610   precision 0.9051
```

En general, las metricas de Gradient Boosting son concretamente mejores en el area del recall y la precision. El modelo de Gradient Bosting tiene una precision de 0.9051 sobre la precision de 0 del modelo de linea base, lo que significa que el modelo de Gradient Boosting es capaz de identificar correctamente los casos de fraude y tiende a identificar una proporcion mayor de casos reales comparados con el total de casos reales y falsos positivos. 

En el caso del *recall* del GB el incremento a 0.4610 sobre el 0 del modelo base indica que el modelo es capaz de identificar corrrectamente casos de fraude con respecto de los falsos negativos .

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

|  | 0 | 1 |
| :--- | :--- | :--- |
| 0 | 5691 | 14 |
| 1 | 159 | 136 |

Con el umbral de 0.5 el modelo de Gradient Boosting deja pasar 159 casos marcados como negativos cuando eran positivos.

El umbral elegido en el paso .5 es de 0.1.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

_(tu respuesta)_

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

_(tu respuesta)_

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

_(tu respuesta)_

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

_(tu respuesta)_

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

_(tu respuesta)_
