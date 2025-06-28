### **Proyecto Final: Sistema de Reconocimiento de Texto en Imágenes y Análisis de Sentimiento**

**Objetivo del Proyecto:**

Desarrollar un sistema de IA de dos etapas que pueda:

1. **Leer texto de una imagen:** Utilizar un modelo entrenado con el dataset **IIIT-5K Words** para detectar y transcribir texto encontrado en una imagen (ej. un cartel, una captura de pantalla, un anuncio).
2. **Analizar su sentimiento:** Tomar el texto extraído y pasarlo a un modelo de Red Neuronal Recurrente (RNN) —entrenado con el **Twitter Sentiment Dataset**— para determinar si su contenido es positivo, negativo o neutral.

El objetivo es crear un pipeline que pueda "ver" una imagen, "leer" lo que dice y "entender" su tono.

---

### **Parte 1: Instrucciones del Proyecto (Paso a Paso)**

Esta guía describe **qué** deben lograr. La investigación sobre **cómo** implementarlo es el núcleo del aprendizaje en este proyecto.

**Fase 0: Planificación y Configuración del Entorno**

1. **Configuración del Repositorio:** Creen un nuevo repositorio en GitHub. Debe incluir un archivo `README.md` y un `.gitignore` apropiado para proyectos de Python y Machine Learning.
2. **Entorno Virtual:** Aísle las dependencias del proyecto configurando un entorno virtual con `venv` o `conda`.
3. **Instalación de Librerías:** Identifiquen las librerías necesarias (ej. TensorFlow/PyTorch, OpenCV, Pandas, Scikit-learn, Scipy para leer los archivos `.mat` del dataset) e instálenlas. Documenten todo en un archivo `requirements.txt`.

**Fase 1: Modelo de Reconocimiento de Texto en Escenas (STR)**

1. **Obtención y Exploración de Datos (IIIT-5K Dataset):**
    - Descarguen el dataset de Kaggle.
    - Investiguen su estructura. Notarán que las etiquetas (el texto de cada imagen) a menudo se encuentran en archivos `.mat` (formato de MATLAB). Pero tambien esta en formato CSV, pueden ignorar los .mat
    - Escriban los scripts para asociar correctamente cada archivo de imagen con su palabra o texto correspondiente. (Tip, el path es una columan en el CSV)
2. **Preprocesamiento de las Imágenes:**
    - Diseñen un pipeline de preprocesamiento. A diferencia del texto manuscrito limpio, las imágenes de escenas pueden tener colores, fondos ruidosos y diferentes tipos de letra. Consideren pasos como: cambio de tamaño, normalización de píxeles, y quizás técnicas de aumento de datos (data augmentation) más avanzadas.
3. **Diseño y Entrenamiento del Modelo STR:**
    - Investiguen arquitecturas de modelos para STR. La combinación de Redes Convolucionales (CNN) para la extracción de características visuales y Redes Recurrentes (RNN) para interpretar la secuencia de caracteres es el enfoque más común y exitoso.
    - Implementen la arquitectura que elijan y la función de pérdida.
    - Entrenen el modelo usando los datos de IIIT-5K.
4. **Evaluación del Modelo STR:**
    - Evalúen el rendimiento del modelo en el conjunto de prueba del dataset. Utilicen métricas estándar como la **Tasa de Error de Caracteres (CER)** y la **Tasa de Error de Palabras (WER)**.

```python
import jiwer
# Ground truth and hypothesis sentences

ground_truth = "this is a test sentence"

hypothesis = "this a sentenc"

# Calculate Word Error Rate (WER)

wer = jiwer.wer(ground_truth, hypothesis)

print(f"Word Error Rate (WER): {wer}")

# Calculate Character Error Rate (CER)

cer = jiwer.cer(ground_truth, hypothesis)
print(f"Character Error Rate (CER): {cer}")
```

---

1. **Aplicación a una Imagen Completa:**
    - El modelo fue entrenado con imágenes de palabras ya recortadas. El siguiente paso es generalizarlo para que funcione en una imagen completa que contenga texto.
    - Implementen un método de **detección de texto**. Recomiendo usar contornos con OpenCV .
    - El objetivo es una función que tome una imagen grande, encuentre las regiones de texto, las recorte y las pase a su modelo de reconocimiento.

**Fase 2: Modelo de Análisis de Sentimiento**

1. **Obtención y Exploración de Datos (Twitter Dataset):**
    - Clonen o descarguen el dataset desde GitHub.
    - Analicen la estructura del CSV, identifiquen las columnas de texto y sentimiento, y exploren la distribución de las clases.
2. **Preprocesamiento del Texto:**
    - Diseñen un pipeline de limpieza para los tuits: convertir a minúsculas, eliminar URLs, @menciones y hashtags, tokenizar y eliminar *stop words*.
    - Conviertan el texto limpio a una representación numérica (secuencias de enteros, embeddings como Word2Vec/GloVe, o embeddings entrenados desde cero).
3. **Diseño y Entrenamiento del Modelo RNN:**
    - Diseñen y construyan una arquitectura RNN para clasificación de texto usando **LSTM** o **GRU**.
    - Entrenen el modelo con los datos de Twitter preprocesados.
4. **Evaluación del Modelo de Sentimiento:**
    - Evalúen el modelo usando **Accuracy**, **Precisión**, **Recall**, **F1-Score** y una **matriz de confusión**.

**Fase 3: Integración y Demostración Final**

1. **Construcción del Pipeline Completo:**
    - Creen un script principal que integre todos los componentes. La entrada debe ser la ruta a una imagen que contenga texto.
2. **Ejecución de Extremo a Extremo:**
    - El pipeline debe ejecutar la siguiente secuencia de forma automática: a. Cargar la imagen de entrada. b. Detectar y recortar las áreas de texto en la imagen. c. Para cada área de texto recortada, usar el modelo para transcribirla. d. Unir todo el texto transcrito en un solo bloque. e. Preprocesar este texto para el análisis de sentimiento. f. Pasarlo al modelo de sentimiento para obtener la predicción final.
    - El sistema debe mostrar como salida el texto completo reconocido y el sentimiento predicho.

---

### **Parte 2: Tips para Manejar Datasets Grandes**

Aunque los datasets de este proyecto no son excesivamente grandes (el de Twitter es el mayor), los siguientes hábitos son cruciales para cualquier proyecto de Machine Learning:

1. **Generadores de Datos (Data Generators):** Es la práctica más importante. En lugar de usar `model.fit(X_train, y_train)`, donde cargas todo en memoria, usa `model.fit(data_generator)`. Un generador (como `tf.data.Dataset` o `torch.utils.data.DataLoader`) carga y procesa los datos del disco en lotes pequeños, manteniendo el uso de RAM bajo y estable.
2. **Procesamiento por Lotes (Batch Processing):** Si necesitas hacer un preprocesamiento pesado una sola vez, hazlo por partes. Escribe un script que lea un subconjunto de archivos, los procese, guarde el resultado y continúe con el siguiente, liberando memoria entre lotes.
3. **Empezar con una Muestra (Subsampling):** Para depurar y experimentar rápidamente, trabaja con una fracción del dataset (5-10%). Esto te permite probar arquitecturas y pipelines sin esperar horas. Una vez que el código es estable, puedes entrenar con los datos completos.
4. **Aprovechar el Cómputo en la Nube:** Utiliza **Google Colab** o **Kaggle Kernels**. Ambos ofrecen acceso gratuito a GPUs (que aceleran el entrenamiento drásticamente) y suelen tener más memoria RAM que un portátil estándar.
5. **Formatos de Archivo Eficientes:** Para datos intermedios (ej. el mapeo imagen-texto ya procesado), considera formatos binarios como **HDF5**, **Parquet** o **TFRecord**. Son más rápidos de leer y ocupan menos espacio que CSVs.
6. **Optimizar Tipos de Datos:** Al usar Pandas, comprueba los `dtypes`. Puedes reducir el uso de memoria cambiando un `int64` a `int32` o un `float64` a `float32` si el rango de tus datos lo permite.
7. **Limpieza Explícita de Memoria:** En notebooks (Jupyter, Colab), las variables grandes pueden quedarse en memoria. Usa `del mi_variable` y luego `import gc; gc.collect()` para forzar al recolector de basura a liberar espacio.

---

### **Parte 3: Criterios de Evaluación**

La evaluación se basará en dos entregables: el **repositorio de código** y un **informe del proyecto**.

**1. Repositorio de Código (en GitHub):**

- **Organización y Claridad (25%):**
    - Estructura de carpetas lógica (`data/`, `notebooks/`, `src/`).
    - `README.md` completo: explica el proyecto, cómo instalar dependencias (`requirements.txt`) y cómo ejecutar el pipeline final con una imagen de ejemplo.
    - Código limpio, comentado y modular (uso de funciones y clases).
- **Funcionalidad y Calidad Técnica (40%):**
    - El código es reproducible y funciona como se describe.
    - Implementación correcta de los modelos STR y RNN.
    - El pipeline de integración (detección + reconocimiento + análisis) funciona de extremo a extremo.
- **Control de Versiones (10%):**
    - Uso adecuado de Git, con `commits` frecuentes y mensajes descriptivos.

**2. Informe del Proyecto (en formato PDF):**

- **Estructura y Presentación (10%):**
    - Documento bien estructurado: Introducción, Metodología, Resultados, Discusión/Análisis, Conclusiones y Referencias.
    - Redacción clara y profesional.
- **Rigor Técnico y Análisis (25%):**
    - Descripción detallada de las arquitecturas de los modelos y justificación de las decisiones de diseño.
    - Explicación completa de los pipelines de preprocesamiento de datos.
    - Presentación clara de los resultados: tablas con métricas (CER, WER, F1-Score), matrices de confusión y gráficos de entrenamiento.
    - **Discusión crítica:** ¿Qué significan los resultados? ¿Cuáles fueron los mayores desafíos (ej. en la detección de texto)? ¿Qué limitaciones tiene el sistema? ¿Cómo podría mejorarse?