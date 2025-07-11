
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
import cv2
import numpy as np

def build_str_model(input_shape, num_classes):
    """
    Construye un modelo STR (Scene Text Recognition) basado en CNN-RNN.
    input_shape: (height, width, channels) de las imágenes preprocesadas.
    num_classes: Número de caracteres únicos + 1 para CTC blank.
    """
    # CNN para extracción de características visuales
    input_img = Input(shape=input_shape, name='image_input')

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv4) # Pool solo en altura para mantener secuencia

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool4 = MaxPooling2D(pool_size=(2, 1))(conv6)

    # Reshape para la capa RNN
    # La salida de la CNN debe ser una secuencia para la RNN
    # (batch_size, width_features, height_features * channels)
    feature_map_shape = pool4.get_shape().as_list()
    reshaped = Reshape((feature_map_shape[1], feature_map_shape[2] * feature_map_shape[3]))(pool4)

    # RNN (LSTM bidireccional) para interpretar la secuencia de caracteres
    blstm1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(reshaped)
    blstm2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(blstm1)

    # Capa de salida con softmax para la clasificación de caracteres
    output = Dense(num_classes, activation='softmax', name='output')(blstm2)

    model = Model(inputs=input_img, outputs=output)
    return model

def preprocess_image_for_str(image_path, target_size=(128, 32)):
    """
    Preprocesa una imagen para el modelo STR.
    target_size: (width, height)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Redimensionar manteniendo la relación de aspecto o rellenando
    # Aquí un simple resize, se puede mejorar
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=-1) # Añadir canal para grayscale
    img = img / 255.0 # Normalizar
    return img

def detect_text_regions(image_path):
    """
    Detecta regiones de texto en una imagen completa usando OpenCV.
    Retorna una lista de imágenes recortadas de las regiones de texto.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar un umbral o adaptativeThreshold para binarizar
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtrar contornos pequeños o muy grandes que no sean texto
        if w > 10 and h > 10 and w < img.shape[1] * 0.8 and h < img.shape[0] * 0.8:
            cropped_img = img[y:y+h, x:x+w]
            text_regions.append(cropped_img)
            
    return text_regions

if __name__ == '__main__':
    print("Ejecutando ejemplos de str_model.py")
    
    # Ejemplo de construcción de modelo
    input_shape = (32, 128, 1) # height, width, channels
    num_classes = 80 # Ejemplo: 26 letras + 10 números + símbolos + blank
    model = build_str_model(input_shape, num_classes)
    model.summary()
    
    # Ejemplo de detección de texto (requiere una imagen de prueba)
    # dummy_image_path = 'data/test_image_with_text.png'
    # if os.path.exists(dummy_image_path):
    #     regions = detect_text_regions(dummy_image_path)
    #     print(f"Detectadas {len(regions)} regiones de texto.")
    # else:
    #     print(f"Crea un archivo {dummy_image_path} para probar la detección de texto.")
