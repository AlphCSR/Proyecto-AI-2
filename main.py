import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import cv2 
import numpy as np
#from src.data_loaders import load_iiit5k_data, load_twitter_sentiment_data, image_data_generator, text_data_generator
# from src.str_model import build_str_model, preprocess_image_for_str, detect_text_regions
from src.sentiment_model import preprocess_text_for_sentiment, predict_sentiment
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
    model_save_dir = 'src'
    model_save_path = os.path.join(model_save_dir, 'sentiment_model.keras')
    if os.path.exists(model_save_path):
        print(f"Loading the model from: {model_save_path}")
        try:
            # Load the model
            sentiment_model = tf.keras.models.load_model(model_save_path)

            print("Model loaded successfully.")

            # You can optionally print a summary of the loaded model to verify
            # loaded_model.summary()

        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            sentiment_model = None # Set to None if loading fails
    else:
        print(f"Error: Model file not found at {model_save_path}. Please ensure the model has been saved.")
        sentiment_model = None # Set to None if file not found
    
    vocab_size = 10000
    sentiment_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")

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