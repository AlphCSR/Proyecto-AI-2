
# Proyecto Final: Sistema de Reconocimiento de Texto y Análisis de Sentimiento

Este repositorio contiene el código para un sistema de IA de dos etapas que lee texto de imágenes y analiza su sentimiento.

## Descripción

El proyecto se divide en dos componentes principales:

1.  **Reconocimiento de Texto en Escenas (STR):** Un modelo basado en CNN y RNN entrenado con el dataset IIIT-5K para transcribir texto de imágenes.
2.  **Análisis de Sentimiento:** Un modelo RNN (LSTM/GRU) entrenado con el Twitter Sentiment Dataset para clasificar el texto extraído como positivo, negativo o neutral.

## Estructura del Repositorio

-   `data/`: Carpeta para almacenar los datasets (no incluida en Git).
-   `notebooks/`: Jupyter Notebooks para exploración y experimentación.
-   `src/`: Scripts de Python para el preprocesamiento, los modelos y el pipeline final.
-   `main.py`: Script principal para ejecutar el pipeline completo.
-   `requirements.txt`: Dependencias del proyecto.

## Cómo Empezar

### 1. Clonar el Repositorio

```bash
git clone <URL-DEL-REPOSITORIO>
cd <NOMBRE-DEL-REPOSITORIO>
```

### 2. Crear y Activar un Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el Pipeline

Para procesar una nueva imagen, usa el siguiente comando:

```bash
python main.py --image_path /ruta/a/tu/imagen.png
```
