import argparse
import os
import cv2 
import numpy as np
from src.data_loaders import load_iiit5k_data, load_twitter_sentiment_data, image_data_generator, text_data_generator
# from src.str_model import build_str_model, preprocess_image_for_str, detect_text_regions
from src.sentiment_model import build_sentiment_model, preprocess_text_for_sentiment, predict_sentiment
import easyocr

# TODO: Definir estos valores basados en el entrenamiento real
STR_MODEL_INPUT_SHAPE = (32, 128, 1) # height, width, channels
STR_NUM_CLASSES = 80 # Ejemplo: 26 letras + 10 números + símbolos + blank

SENTIMENT_VOCAB_SIZE = 10000 # Tamaño del vocabulario
SENTIMENT_EMBEDDING_DIM = 100
SENTIMENT_MAX_SEQUENCE_LENGTH = 100 # Longitud máxima de las secuencias
SENTIMENT_NUM_CLASSES = 3 # Negativo, Neutral, Positivo

global_reader = easyocr.Reader(['en', 'es'])

def main(image_path):
    """
    Pipeline principal para el reconocimiento de texto y análisis de sentimiento.
    """
    print(f"Procesando la imagen: {image_path}")

    # 1. Cargar modelos (esto debería hacerse una vez al inicio de la aplicación real)
    # sentiment_model = build_sentiment_model(SENTIMENT_VOCAB_SIZE, SENTIMENT_EMBEDDING_DIM, SENTIMENT_MAX_SEQUENCE_LENGTH, SENTIMENT_NUM_CLASSES)
    # TODO: Cargar pesos pre-entrenados: str_model.load_weights('path/to/str_weights.h5')
    # TODO: Cargar pesos pre-entrenados: sentiment_model.load_weights('path/to/sentiment_weights.h5')
    # TODO: Cargar tokenizer de sentimiento: sentiment_tokenizer = load_tokenizer('path/to/tokenizer.pkl')

    # Placeholder para modelos y tokenizer cargados
    sentiment_model = None # Reemplazar con el modelo cargado
    sentiment_tokenizer = None # Reemplazar con el tokenizer cargado

    # Cargar la imagen usando OpenCV para una gestión de errores más robusta
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen desde {image_path}. Verifica la ruta y el formato del archivo.")
            return
        print(f"Imagen cargada exitosamente. Dimensiones: {image.shape}")
    except Exception as e:
        print(f"Error al cargar la imagen con OpenCV: {e}")
        return

    print("Detectando y transcribiendo texto con EasyOCR...")
    try:
        # EasyOCR hace todo el trabajo de detección y reconocimiento
        results = global_reader.readtext(image)
        print(f"Detectadas {len(results)} entradas de texto.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error al procesar la imagen con EasyOCR: {e}")
        return

    full_text_recognized_parts = []
    if not results: # Si EasyOCR no encontró ningún resultado
        print("No se detectó texto en la imagen.")
        return

    # Extraer el texto de los resultados de EasyOCR
    for (bbox, text, conf) in results:
        if text: # Asegurarse de que el texto no esté vacío
            full_text_recognized_parts.append(text)

    # Unir todo el texto reconocido en un solo bloque
    final_text = " ".join(full_text_recognized_parts)

    if not final_text.strip():
        print("No se pudo extraer texto significativo de la imagen.")
        return

    print(f"Texto reconocido completo: \"{final_text}\"")





    # 3. Preprocesar este texto para el análisis de sentimiento.
    print("Preprocesando texto para análisis de sentimiento...")
    processed_sentiment_text = preprocess_text_for_sentiment(final_text)
    print(f"Texto preprocesado para sentimiento: \"{processed_sentiment_text}\"")

    # 4. Pasarlo al modelo de sentimiento para obtener la predicción final.
    if sentiment_model and sentiment_tokenizer:
        print("Analizando sentimiento...")
        predicted_sentiment, probabilities = predict_sentiment(sentiment_model, sentiment_tokenizer, processed_sentiment_text, SENTIMENT_MAX_SEQUENCE_LENGTH)
        print(f"Sentimiento predicho: {predicted_sentiment} (Probabilidades: {probabilities})")
    else:
        print("Modelos de sentimiento no cargados. Saltando análisis de sentimiento.")
        print("Para habilitar el análisis de sentimiento, asegúrate de cargar los modelos y el tokenizer.")

    # 5. Mostrar resultados (ya se hace en los pasos anteriores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza el texto de una imagen y determina su sentimiento.")
    parser.add_argument("--image_path", type=str, required=True, help="Ruta a la imagen de entrada.")
    args = parser.parse_args()

    main(args.image_path)