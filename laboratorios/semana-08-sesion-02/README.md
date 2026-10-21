# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas ya lo tienes desde la Semana 4. Instala matplotlib:

```
pip install matplotlib
```

Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. También puedes correr el mismo código en un notebook de Jupyter, donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Segun el articulo, la idea de usar EDA es responder a la gran pregunta de *¿qué es lo que tengo?*, esta gran 
pregunta se responde mediante el análisis de los valores de las variables, valores atípicos, distribuciones de las 
variables, tipos de datos, valores faltantes, correlaciones, etc. Para entender que datos tenemos realizamos un 
análisis numérico y estadístico sobre la data y esto es lo que responde el EDA antes de entrenar cualquier modelo.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para el análisis de multiples variables o entre varias, el artículo propone el uso de un análisis de correlación y 
el análisis de interacciones. En un caso analizamos la tendencia de una variable frente a la otra y como esta varía, 
basándose en correlaciones de pearson o spearman que determinan como las variables y sus valores fluctuan juntas, 
crecimiento, decrecimiento, etc. Las interacciones analizan como actuan un par de variables, esto representa un 
análisis de variables y su comportamiento frente a otras de manera visual, mientras que la correlación traduce este 
análisis a una cantidad numérica.

Para el análisis univariable, el articulo distingue entre dos tipos de variables. Para las variables categóricas, 
propone el uso de gráficos de frecuencia, y el análisis de la moda de estas variables. Para las variables numéricas, 
propone el uso de histogramas, boxplots y el análisis de la estadística descriptiva por variable, así como el 
análisis de valores faltantes y outliers.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

Notaríamos, tal vez, dentro de las clasificaciones del modelo que los valores resultantes salen muy por encima del 
promedio de las tarifas reales registradas, esto lo notaríamos a través de los logs de los resultados de predicción, 
lo que llevaría a que notaramos la discrepancia y a que consideraramos el reentreno del modelo. Sin embargo, si solo miramos el resultado final del modelo, no notaríamos nada, ya que el modelo
por si solo no presenta nada de información sobre si la tarifa en el contexto real es correcta o no, el modelo 
podría tener un buen desempeño en métricas pero tener estas clasificaciones atípicas. La única forma, entonces, de 
notar esto en un modelo en producción es a través del monitoreo del mismo, o en desarrollo mediante el conocimiento del
dataset y el entorno del modelo.

## Ejercicio 1: resumen, distribución y atípicos

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv) tiene 3000 viajes en taxi de Nueva York: fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Primera mirada: qué contiene, cómo se distribuye la tarifa y qué valores están fuera de lo esperable.

**Respuesta 1, antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

Una tarífa de cero o una distancia negativa, ya que no es posible que un viaje tenga una tarifa de cero o que la 
distancia sea negativa sin que haya por detrás un proceso de error que puede haber resultado en que todos los datos 
registrados esten en error. Esto indicaría un error en la captura de datos.

**Respuesta 2:** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No, no todos los viajes pueden ser un error, hay que notar que los rangos intercuartiles son medidas estadísticas 
que deben ser evaluadas en el dataset y con su contexto, es posible que los outliers superen el Q3 que corresponde a 
100$ y que la tarifa de esos viajes a eliminar sea de 150$ es tecnicamente posible, pero si nos basamos solo en 
el IQR, estos viajes serían considerados outliers. Para averiguar si son errores o no, habría que analizar la 
distribución de los valores por encima de los IQRs, en el rango intercuartil y en cada segmento, incluyendo los 
valores negativos. Es posible así encontrar la distribución real de valores e identificar si los valores sobre el 
IQR realmente son errores reales o son simplemente viajes más largos.

**La formulación completa, tal como la resuelvo yo:**

```python
def explore_initial_dataset_summary_and_info(df: pd.DataFrame) -> None:
    print("Exploring initial dataset")
    print("Filas y columnas:", df.shape)
    print(f"Tipos de datos:\n{df.dtypes}")
    print(f"Valores nulos por columna:\n{df.isna().sum()}")
    print(f"Resumen estadístico:\n{df.describe().round(2)}")
```

```console
Exploring initial dataset
Filas y columnas: (3000, 10)
Tipos de datos:
id_viaje               int64
fecha_hora_inicio     object
zona_origen           object
zona_destino          object
distancia_km         float64
duracion_min         float64
pasajeros            float64
metodo_pago           object
tarifa               float64
propina              float64
dtype: object
Valores nulos por columna:
id_viaje               0
fecha_hora_inicio      0
zona_origen            0
zona_destino           0
distancia_km           0
duracion_min           0
pasajeros            120
metodo_pago           60
tarifa                 0
propina                0
dtype: int64
Resumen estadístico:
       id_viaje  distancia_km  duracion_min  pasajeros   tarifa  propina
count   3000.00       3000.00       3000.00    2880.00  3000.00  3000.00
mean    1500.50          8.44         24.66       1.61    31.39     3.29
std      866.17          7.16         18.80       1.15    70.99     3.63
min        1.00          0.00        -74.00       0.00     1.60     0.00
25%      750.75          3.74         12.10       1.00    13.99     0.00
50%     1500.50          6.73         20.45       1.00    22.60     2.48
75%     2250.25         10.51         30.70       2.00    32.55     4.79
max     3000.00         45.43        151.40       6.00   999.99    17.49
Atipicos de tarifa segun IQR con df selection: 334
Atipicos de tarifa segun IQR con query: 334
```


## Ejercicio 2: relaciones entre variables

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/explorar_relaciones.py`](./plantilla-ejercicio2/explorar_relaciones.py) y córrela desde su carpeta (`cd plantilla-ejercicio2`, `python3 explorar_relaciones.py`): lee el mismo `starter/viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

En este caso el gráfico siguiente demuestra que hay un grupo de valores de distancia cero que tuvieron una tarifa 
positiva, 25 casos muestran esta relación. Esto indica un error en el registro de los datos, o casos especiales como 
fallas en el medidor o simplemente casos en donde el operador olvido apagar el medidor y registro una tarifa 
positiva cuando el vehículo no estaba en movimiento.

![](/Users/santiagoarellano/Documents/RandomGits/data-science-end-to-end/laboratorios/semana-08-sesion-02/starter/figuras/distancia_vs_tarifa.png)


Además, el gráfico demuestra una relación extraña en donde hay un valor de tarifa de cerca de 70 dólares recurrente 
en varias distancias registradas.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

```console
Propina promedio y cantidad de viajes con propina 0 por metodo de pago:
  metodo_pago  propina_promedio  cantidad_viajes_con_propina_0
0    efectivo          0.000000                            888
1     tarjeta          4.728762                              0
```

En este caso, el análisis demuestra que en los pagos con efectivo es más probable registrar un viaje con propina 0 
que en los pagos con tarjeta de crédito. Esto puede deberse a que los pasajeros que pagan con tarjeta de crédito son 
más propensos a dejar propina, o a que el sistema de registro de pagos con tarjeta de crédito no permite registrar 
una propina de 0, mientras que el sistema de efectivo sí lo permite.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que 
estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

```console
Correlación de las variables con la tarifa:
distancia_km    0.96
duracion_min    0.96
pasajeros      -0.05
propina         0.63
dtype: float64
```

En este caso la duración tiene una correlación alta al giual que la distancia con la tarifa, esto es verdad. La 
distancia no se puede usar al momento que alguien pida un taxi dado que no es un factor conocido hasta el final del 
recorrido, si la distancia se conociera desde un inicio, como en una aplicación de Uber, evaluaría su utilización en 
base de la importancia en el modelo y si contar con esta variable en el entrenamiento realmente apoya. Esta variable 
para predicciones en tiempo real no es útil, porque no se conocen y el modelo funciona en base al conocimiento completo
de un viaje, incluyendo la distancia total y la duración total. La duración tampoco se puede usar directamente 
porque usarla así terminaría en un modelo que predice con información parcial. Podría usarse para obtener, por ejemplo, 
una aproximación al posible costo final y ajustar al modelo en base de la calidad de las predicciones iniciales a 
medida que se obtiene más información y finalmente el resultado real de la predicción.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

_(filas, columnas, faltantes por columna, y cuántos atípicos marcó el IQR en tu objetivo)_

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

_(qué gráfico o consulta hiciste, y qué mostró)_

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1:**

_(tu hallazgo)_

**Hallazgo 2:**

_(tu hallazgo)_

**Hallazgo 3:**

_(tu hallazgo)_

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

_(tu respuesta)_
