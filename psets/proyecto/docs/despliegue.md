# Estrategia de despliegue del modelo

**Patrón seleccionado: inferencia por lotes (*batch inference*) con ejecución diaria.**

El proyecto implementa un modelo de clasificación binaria supervisada para estimar la probabilidad de falla de unidades SSD del modelo MC1 (modelo con la mayor cantidad de casos de error en el dataset de entrenamiento) durante los próximos 30 días luego del registro de telemetría SMART de ese día. Se selecciona un esquema de inferencia por lotes debido a la naturaleza temporal de los datos SMART y al objetivo operacional del sistema, donde se tiene como objetivo priorizar intervenciones preventivas en lugar de responder automáticamente a incidentes en tiempo real.

La ejecución se realizará mediante un trabajo programado diariamente, encargado de recopilar la telemetría SMART de las unidades monitoreadas utilizando herramientas compatibles con sus fabricantes. Posteriormente, los registros serán procesados para generar las características requeridas por el modelo y producir una clasificación de riesgo para cada SSD elegible.

### Decisión habilitada y tiempo de respuesta

Las predicciones permiten construir una lista priorizada de unidades SSD según su probabilidad estimada de falla dentro de los próximos 30 días. Cuando dicha probabilidad supera un umbral definido durante la validación, el dispositivo se considera candidato para revisión preventiva, migración de datos, o reemplazo programado.

Los resultados están dirigidos al equipo de operaciones de almacenamiento y Site Reliability Engineering (SRE), responsable de verificar la criticidad del dispositivo, su redundancia y la viabilidad de una intervención antes de ejecutar cualquier acción.

Se establece una **frecuencia de actualización de 24 horas**, considerando que el sistema busca anticipar fallas con un margen mínimo de siete días para permitir el mantenimiento planificado. No se requiere una respuesta en milisegundos ni inferencia por solicitud individual, puesto que las decisiones de mantenimiento no son automáticas.

### Cantidad de casos y conocimiento anticipado

La unidad de predicción corresponde a cada SSD activo del modelo MC1 en una fecha determinada, identificado mediante `DISK_ID` y `OBSERVATION_DATE`. Aunque el problema de clasificación contempla dos resultados posibles (`TARGET_30D = 0` y `TARGET_30D = 1`), la cantidad de casos que deben procesarse depende del número de unidades elegibles presentes en el inventario operativo.

Estos casos pueden identificarse antes de cada ejecución mediante el inventario de dispositivos y los registros de telemetría disponibles. Por tanto, el conjunto de unidades a evaluar es conocido al iniciar cada lote, aunque su composición puede variar diariamente por incorporaciones, fallas o retiros de dispositivos.

El modelo generará una probabilidad de falla por unidad elegible en cada ejecución, permitiendo procesar conjuntamente el conjunto de dispositivos monitoreados, sin necesidad de mantener un servicio de inferencia permanentemente disponible.

### Disponibilidad de datos durante la inferencia

El modelo requiere la telemetría SMART correspondiente al día de evaluación y el historial de observaciones de cada SSD para calcular las características temporales definidas en la OBT, incluyendo promedios, variaciones e indicadores de disponibilidad de datos en ventanas de 7, 14 y 30 días.

La telemetría será recopilada diariamente mediante un sistema auxiliar con acceso a los dispositivos monitoreados, utilizando herramientas compatibles con los fabricantes. Los registros obtenidos se almacenarán y procesarán mediante el pipeline ETL antes de ejecutar la inferencia por lotes, permitiendo combinar las mediciones actuales con el historial disponible.

Por tanto, **los datos necesarios estarán disponibles al momento de la predicción siempre que la recolección y el procesamiento previos se completen correctamente**. En caso de observaciones faltantes o historial insuficiente, se aplicarán los indicadores de validez y las políticas de tratamiento de valores ausentes definidas durante la preparación de datos, sin asumir que todas las características temporales estarán completas.

### Limitaciones del patrón descartado

Al descartar la inferencia en línea, el sistema pierde la capacidad de actualizar inmediatamente las predicciones ante cambios repentinos en la telemetría SMART de los SSD monitoreados. Bajo el esquema de procesamiento por lotes, las variaciones registradas después de una ejecución serán incorporadas en la siguiente actualización diaria, introduciendo una latencia de hasta 24 horas en la evaluación del riesgo, siempre que el procesamiento se complete según lo programado.

Esta limitación es aceptable dentro del alcance del proyecto, puesto que el modelo utiliza observaciones y características temporales de resolución diaria para estimar fallas en un horizonte de 30 días, buscando una anticipación mínima de siete días que permita planificar intervenciones preventivas.

El sistema no pretende sustituir los mecanismos existentes de monitoreo y alerta inmediata, sino complementarlos mediante una clasificación periódica de riesgo que facilite la priorización del mantenimiento.
