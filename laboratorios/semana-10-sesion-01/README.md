# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje es una técnica de aprendizaje automático que combina las predicciones de varios modelos para obtener 
un resultado más preciso y estable. En bagging, los modelos se entrenan de manera independiente utilizando 
diferentes muestras aleatorias de los datos, obtenidas con reemplazo. Sus predicciones se combinan mediante una 
votación mayoritaria o un promedio. En cambio, boosting entrena los modelos de manera secuencial, donde cada nuevo 
modelo se construye considerando los errores de los anteriores. Además, l  contribución de cada modelo al resultado 
final puede depender de su desempeño. La principal diferencia se encuentra en que bagging utiliza modelos 
independientes, mientras que boosting construye un conjunto de modelos que se complementan progresivamente.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Un promedio de varios árboles puede producir mejores predicciones porque cada árbol aprende patrones diferentes y 
puede cometer errores distintos. Si estos errores no ocurren siempre de la misma manera, al combinar sus 
predicciones algunos pueden compensarse. Esto permite reducir la variabilidad de los resultados y evitar que las 
decisiones de un solo árbol tengan demasiada influencia. Sin embargo, la mejora depende de que exista suficiente 
diversidad entre los árboles, ya que combinar modelos que cometen los mismos errores no necesariamente mejora las 
predicciones.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_

## Ejercicio 1: línea base, un árbol y un bosque

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Esperariamos que tenga una precision del 50%, es simplemente un modelo que nos sirve como referencia para comparar 
otros modelos. Si un modelo no puede superar la línea base, entonces no es útil.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

Esperariamos un error muy bajo en entrenamiento, porque tenderia a realizar overiftting a los propios datos de 
entrenamiento y por ende durante las pruebas mostraria un error mucho mayor, ya que no generaliza bien a datos nuevos.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad del árbol de decisión, se espera que disminuya la brecha entre los errores de 
entrenamiento y prueba, ya que el modelo tendrá menos capacidad para aprender patrones demasiado específicos de los 
datos. Esto se confirma en los resultados obtenidos, donde el árbol sin restricciones presenta una brecha de 1.6988, 
mientras que el árbol con max_depth=8 reduce esa diferencia a 0.2630. Aunque el error de entrenamiento aumenta de 0.
0029 a 1.0098, el error de prueba disminuye de 1.7017 a 1.2728, lo que indica una mejor capacidad de generalización.

Por otro lado, al utilizar un RandomForestRegressor con 200 árboles, se espera reducir la variabilidad de las 
predicciones mediante el promedio de los resultados de varios árboles entrenados con diferentes muestras. En la 
ejecución, este modelo obtuvo un error de entrenamiento de 0.4743 y un error de prueba de 1.2989, con una brecha de 
0.8245. Aunque presenta una mejora frente al árbol sin restricciones, su brecha es mayor que la del árbol limitado a 
ocho niveles. Por lo tanto, en esta ejecución, el árbol con profundidad limitada obtuvo el menor error de prueba y 
la menor brecha entre entrenamiento y prueba de los tres modelos basados en árboles.

**Lo que confirmó la tabla, TODO por TODO:**

```console
Dummy Regressor              entrenamiento 13.1998   prueba 12.5308
Decision Tree Regressor      entrenamiento  0.0029   prueba  1.7017
Decision Tree Regressor (max_depth=8) entrenamiento  1.0098   prueba  1.2728
Random Forest Regressor      entrenamiento  0.4743   prueba  1.2989

Resultados finales:
                                  modelo  mae_entrenamiento  ...  brecha  segundos
0                        Dummy Regressor            13.1998  ... -0.6690     0.006
1                Decision Tree Regressor             0.0029  ...  1.6988     0.014
2  Decision Tree Regressor (max_depth=8)             1.0098  ...  0.2630     0.010
3                Random Forest Regressor             0.4743  ...  0.8245     1.192

[4 rows x 5 columns]
```

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

En terminos de columnas, dado que es el mismo mecanimmo de implementaciones pasadas, la regression con interacciones 
de 3 grados genera, desde 25 columnas finales luego del scaling y encoding, 3275 columnas finales para el 
entrenamiento del modelo. En filas, el archivo de entrenamiento se cuenta con 2975 filas. La comparación entre el 
número de columnas y filas es importante porque cuando el número de columnas es mayor que el número de filas, el 
modelo tiende a sobreajustarse a los datos de entrenamiento, lo que resulta en un error de prueba más alto, ya que 
el modelo no generaliza bien a datos nuevos. 

En este caso si bien la brecha final no es elevada como se esparaba en la cmoprobacion del archivo original, una 
brecha de 22 puntos de error en pruba demuestra que el modelo no está generalizando bien a los datos de prueba, lo 
que indica un sobreajuste.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En este caso la diferencia clave es que al imponer un sistema de regualizacion de las features ingresadas al sistema,
es posible reducir el peso que tienen varias de estas y por tanto hacer un pruning de las features que no aportan al 
modelo, reduciendo la posibilidad de overfitting al disminuir la complejidad del modelo. Esto se refleja en el error 
de prueba, que disminuye significativamente al pasar de un modelo lineal grado 3 a un modelo Ridge grado 3.

En el error de entrenamiento el modelo tiene un mayor error en el caso de Ridge junto con Polynomial features porque 
el modelo deja de sobreajustarse a los datos de entrenamiento, lo que resulta en un error mas alto. Sin embargo, 
vemos una historia completamente diferente en prueba, donde el modelo con Ridge tiene un error de prueba de 1.32 
comparado con el 22.98 de la regresión lineal grado 3, lo que indica que Ridge ha mejorado significativamente la 
capacidad de generalización del modelo.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

El modelo con error de prueba mas bajo corresponde a Gradient Boosting, pero el error de entrenamiento mas bajo se 
da en Ridge Grade 3. Ambos modelos tienen los errores mas bajos de todos los modelos probados.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

La brecha negativa ocurre porque el modelo `DummyRegressor` utiliza la mediana de las tarifas del conjunto de 
entrenamiento para realizar una predicción constante, sin considerar las características de cada viaje. Aunque esta 
mediana minimiza el error absoluto dentro del entrenamiento para una predicción constante, no garantiza que los 
errores de prueba sean mayores. En este caso, las tarifas del conjunto de prueba pueden encontrarse, en promedio, 
más cerca de la mediana aprendida, por lo que el error absoluto medio resulta menor. Esto explica que se obtenga un 
error de 13.1998 en entrenamiento y 12.5308 en prueba, con una brecha de -0.6690. La diferencia no significa que el 
modelo haya aprendido mejores patrones, sino que los dos conjuntos contienen observaciones con distribuciones 
ligeramente diferentes.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.

### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

_(tu respuesta)_

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

_(tu respuesta)_

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

_(tu respuesta)_

**¿Cuánto mejora tu mejor modelo a la línea base?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_
