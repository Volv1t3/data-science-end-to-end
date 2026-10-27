# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

La fuga es un proceso de contaminacion que se da cuando un modelo aprende, durante su entrenamiento, de informacion 
que no deberia de tener acceso a, como por ejemplo datos del conjunto de prueba, o variables *proxy* que representen 
o informen al modelo del resultado final de la variable objetivo. Esto hace que un modelo presente un overfitting al
conjunto de prueba, y ademas se manifiesta en un model que aprende de los patrones de los datos en los que se evalua,
por lo que siempre tendra resultados altos en sus metricas. 

En consecuencia, cuando el modelo llegue a produccion se encontrara con datos que no ha visto antes, y que tal vez 
no tengan nada de similar con los datos a los que hizo overfitting durante el entrenamiento. Si el modelo aprendio 
los patrones de los datos de prueba, y no aprendio a generalizar sus predicciones, entonces tendra resultados bajos
comparado con data real.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

Este es un ejemplo de una variable *proxy*, que es una variable que representa o informa al modelo del resultado 
final de la variable objetivo. En este caso, la duración del viaje es una variable que tiene una correlacion muy 
alta con el precio del viaje, y por lo tanto, el modelo aprende a predecir el precio del viaje a partir de la 
duracion del mismo, en lugar de aprender a generalizar sus predicciones a partir de las variables que realmente 
deberian influir en el precio del viaje.

En este caso, la duracion no es una variable que se pueda conocer al iniciar un proceso de prediccion. SI el modelo 
se usara para predecir el precio futuro de una carrera de taxi, antes de que esta inicie, es imposible que el modelo 
conozca la duracion final del viaje para usar esa feature, no obstante, si conoces la duracion del viaje es posible 
que el modelo aprenda a derivar el precio de la distancia y la duracion, y por lo tanto, el modelo no aprende a 
generalizar sus predicciones a partir de las variables que realmente puede receptar durante el inicio de una carrera.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Si calculas con todos los valores tendras un dataset en donde todos los valores se escalan con el mismo conjunto, 
incluyendo los valores de entrenamiento y prueba, generando data leakage. El escalamiento, o preprocesamiento se 
realiza siempre con un subconjunto, el subconjunto de entrenamiento, y luego se aplica a los datos de prueba.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

```python
X_preparado = escalador.fit_transform(imputador.fit_transform(X)) #Imputing with the entire data set instead of the
# partition
```

En la seccion del escalador, cuando se usa todo el connjunto para el fitting del imputador y luego del escalador.

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

En este aso las 2975 no son un error en si, sino que es una validacion que se hace para detectar que el conjunto en 
su totalidad que el escalador vio durante su fit y transform. En este caso notamos que ambas implementaciones muestran 
que el escalador vio 2975 filas, lo que indica que el conjunto de prueba se incluyo en el fit del escalador, y por 
lo tanto se esta cometiendo data leakage.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

En este caso no necesariamente la diferencia tiene que ser grande. En mi caso la ejecucion de los dos archivos, mi 
corregido y el de la clase da una diferencia de 0.0002 entre ambas mediciones, y el mio si implementa el mecanismo 
de validacion de que el de la clase no hace para separar la implementacion. Las medianas de los datos convergen en 
ambas implementaciones a una distania de 0.001. Aunque la fuga existe, la implementacion es la que varia y resulta 
en diferentes errores en los modelos, si mejoramos el proceso de entrenamiento veremos variaciones, pero no 
deberiamos de ver variaciones elevadas en los resultados si lo unico que cambia es la distribucion de los datos.

**Lo que confirmó la salida, TODO por TODO:**

_(anota las filas que vio cada escalador y los dos errores)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Después del preprocesamiento se obtuvieron **25 columnas a partir de las 10 características originales**. Las dos 
variables numéricas conservaron sus columnas después de la imputación y el escalado, mientras que las cinco 
variables previamente preparadas permanecieron sin modificaciones. Las tres variables categóricas se transformaron 
mediante `OneHotEncoder`, generando ocho columnas para `zona_origen`, ocho para `zona_destino` y dos para 
`metodo_pago`. Por lo tanto, las 18 columnas binarias generadas, junto con las dos numéricas y las cinco previamente 
preparadas, forman las 25 columnas finales.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Las medianas del imputador conidicen con las del conjunto de entrenamiento, pero no varian mucho con las del 
conjunto completo. Esto demuestra que el imputador se ajustó correctamente solo a los datos de entrenamiento, evitando la
    fuga de información y asegurando que las transformaciones se basen únicamente en los datos disponibles durante el entrenamiento.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

pd.GetDummies hubiese creado una nueva columna para ese viaje en lugar de usar las que ya existen en el transformador. 
COmo el encoding aprende de las clases de entrenamiento, y Bronx no forma parte de las columnas de entrenamiento, 
esta columna no existe en la estructura y por ende mantiene la concordancia con el set de entrenamiento.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

_(tu clasificación)_

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

_(tu decisión)_

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

_(lo que imprimió tu pipeline)_

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_
