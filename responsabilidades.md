 1. Plan de Desarrollo 

  Sprint 1: Fundación y Preparación de Datos
   * Tarea 1: Configuración del Repositorio y Entorno.
       * Crear repositorio en GitHub con README.md, .gitignore.
       * Configurar entorno virtual (venv o conda).
       * Crear y mantener el archivo requirements.txt.
   * Tarea 2: Descarga y Exploración de Datasets.
       * Descargar IIIT-5K Words y Twitter Sentiment Dataset.
       * Analizar la estructura de archivos, columnas (CSV) y formatos.
   * Tarea 3: Scripts de Carga y Asociación de Datos.
       * Desarrollar un script en src/ para leer el CSV de IIIT-5K y asociar cada imagen con su etiqueta de texto.
       * Desarrollar un script para cargar y limpiar el dataset de Twitter.
       * Implementar Data Generators (ej. con tf.data o torch.utils.data.DataLoader) para ambos datasets para no cargar todo en memoria.


  Sprint 2: Modelo de Reconocimiento de Texto (STR)
   * Tarea 4: Preprocesamiento de Imágenes (IIIT-5K).
       * Crear un pipeline de preprocesamiento: cambio de tamaño, normalización, y data augmentation si es necesario.
   * Tarea 5: Diseño e Implementación del Modelo STR (CNN+RNN).
       * Investigar y elegir una arquitectura CNN+RNN.
       * Implementar el modelo como una clase o conjunto de funciones en src/.
   * Tarea 6: Entrenamiento y Evaluación del Modelo STR.
       * Entrenar el modelo usando el generador de datos.
       * Evaluar con las métricas CER y WER. Guardar los resultados.


  Sprint 3: Detección de Texto y Modelo de Sentimiento
   * Tarea 7: Implementación de Detección de Texto.
       * Usar OpenCV para encontrar contornos de texto en imágenes completas.
       * Crear una función que tome una imagen y devuelva una lista de imágenes recortadas (las regiones de texto).
   * Tarea 8: Preprocesamiento de Texto (Twitter Dataset).
       * Crear un pipeline de limpieza de texto: minúsculas, eliminar URLs, menciones, tokenizar, etc.
       * Convertir texto a secuencias numéricas para el modelo RNN.
   * Tarea 9: Diseño e Implementación del Modelo de Sentimiento (RNN).
       * Diseñar e implementar una arquitectura RNN (LSTM o GRU) para clasificación.
   * Tarea 10: Entrenamiento y Evaluación del Modelo de Sentimiento.
       * Entrenar el modelo con los datos de Twitter.
       * Evaluar con Accuracy, Precisión, Recall, F1-Score y matriz de confusión.


  Sprint 4: Integración, Pruebas y Documentación
   * Tarea 11: Creación del Pipeline de Integración.
       * Crear un script main.py que una todo: carga de imagen -> detección de texto -> reconocimiento de texto (STR) -> análisis de sentimiento.
   * Tarea 12: Pruebas End-to-End y Refinamiento.
       * Probar el pipeline completo con imágenes de ejemplo y ajustar los componentes si es necesario.
   * Tarea 13: Redacción del Informe Final.
       * Documentar la metodología, arquitecturas, resultados y conclusiones en un PDF.
   * Tarea 14: Limpieza y Documentación Final del Repositorio.
       * Asegurarse de que el README.md esté completo y explique cómo ejecutar el proyecto.

  ---

  2. Distribución de Tareas (Equipo de 4 Personas)


   * Persona 1: Arquitecto/a de Modelos de Visión
       * Responsabilidades: Tareas 4, 5 y 7.
       * Enfoque: Se concentra en toda la parte de visión por computadora. Es responsable del preprocesamiento de las imágenes, la construcción del modelo STR (CNN+RNN) y la implementación del algoritmo de detección
         de texto con OpenCV.


   * Persona 2: Especialista en NLP y Sentimiento
       * Responsabilidades: Tareas 8, 9 y 10.
       * Enfoque: Se adueña del pipeline de Procesamiento de Lenguaje Natural. Es responsable de limpiar los datos de Twitter, construir y entrenar el modelo RNN de sentimiento, y evaluarlo correctamente.


   * Persona 3: Ingeniero/a de Datos y Pipeline
       * Responsabilidades: Tareas 3 y 11.
       * Enfoque: Es el pegamento del equipo. Se encarga de la ingesta de datos, creando los scripts y generadores para que los modelos puedan entrenar eficientemente. Al final, integra el trabajo de todos en el pipeline final. También gestiona el requirements.txt.


   * Persona 4: Líder de Proyecto y Calidad
       * Responsabilidades: Tareas 1, 2, 6, 12, 13 y 14.
       * Enfoque: Asegura que el proyecto cumpla con los objetivos y la calidad esperada. Configura el repositorio, supervisa las evaluaciones de ambos modelos (ayudando a Persona 1 y 2), lidera las pruebas de integración y es responsable de la documentación final (README e informe).

  Nota: La colaboración es clave. Por ejemplo, Persona 1 y 3 trabajarán juntos en el generador de datos de imágenes, mientras que Persona 2 y 3 lo harán para el de texto.

