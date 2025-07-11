import pandas as pd
import numpy as np
import os
# from scipy.io import loadmat # Para .mat files, si se decide usar

def load_iiit5k_data(csv_path, images_base_path):
    """
    Carga el dataset IIIT-5K Words desde un CSV y asocia imágenes con etiquetas.
    Retorna un generador o una lista de (ruta_imagen, etiqueta).
    """
    print(f"Cargando datos de IIIT-5K desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Ejemplo de cómo asociar: asumiendo que 'path' es la columna con la ruta relativa
    # y 'word' es la etiqueta.
    # Asegúrate de que images_base_path sea la raíz donde están las imágenes.
    
    data_pairs = []
    for index, row in df.iterrows():
        image_full_path = os.path.join(images_base_path, row['path'])
        data_pairs.append((image_full_path, row['word']))
    
    print(f"Cargadas {len(data_pairs)} pares de imagen-texto para IIIT-5K.")
    return data_pairs # Esto debería ser un generador para datasets grandes

def load_twitter_sentiment_data(csv_path):
    """
    Carga el dataset de Twitter Sentiment desde un CSV.
    Retorna un generador o un DataFrame/lista de (texto, sentimiento).
    """
    print(f"Cargando datos de sentimiento de Twitter desde: {csv_path}")
    df = pd.read_csv(csv_path, encoding='latin-1') # O la codificación correcta
    
    # Asume columnas 'text' y 'sentiment'
    # TODO: Limpieza inicial de NaN/duplicados
    
    print(f"Cargados {len(df)} tuits para análisis de sentimiento.")
    return df # Esto debería ser un generador para datasets grandes

def image_data_generator(data_pairs, batch_size, preprocess_fn):
    """
    Generador de datos para imágenes (IIIT-5K).
    Carga imágenes, aplica preprocesamiento y las devuelve en lotes.
    """
    num_samples = len(data_pairs)
    while True:
        # Shuffle data for each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                image_path, label = data_pairs[idx]
                # TODO: Cargar imagen (ej. con OpenCV o PIL)
                # image = cv2.imread(image_path)
                # image = preprocess_fn(image)
                # batch_images.append(image)
                batch_labels.append(label) # Las etiquetas también necesitan preprocesamiento para el modelo STR
            
            # yield np.array(batch_images), np.array(batch_labels)
            yield "placeholder_images", "placeholder_labels" # Placeholder
            
def text_data_generator(dataframe, batch_size, preprocess_fn):
    """
    Generador de datos para texto (Twitter Sentiment).
    Carga texto, aplica preprocesamiento y las devuelve en lotes.
    """
    num_samples = len(dataframe)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_texts = []
            batch_sentiments = []
            
            for idx in batch_indices:
                text = dataframe.iloc[idx]['text']
                sentiment = dataframe.iloc[idx]['sentiment']
                # processed_text = preprocess_fn(text)
                # batch_texts.append(processed_text)
                batch_sentiments.append(sentiment)
            
            # yield np.array(batch_texts), np.array(batch_sentiments)
            yield "placeholder_texts", "placeholder_sentiments" # Placeholder

if __name__ == '__main__':
    # Ejemplo de uso (esto no se ejecutará en el pipeline principal)
    print("Ejecutando ejemplos de data_loaders.py")
    
    # Simular paths
    dummy_iiit5k_csv = 'data/iiit5k_dummy.csv'
    dummy_twitter_csv = 'data/twitter_dummy.csv'
    dummy_images_base_path = 'data/iiit5k_images/'
    
    # Crear archivos dummy para prueba
    os.makedirs('data/iiit5k_images', exist_ok=True)
    with open(dummy_iiit5k_csv, 'w') as f:
        f.write('path,word\n')
        f.write('img1.jpg,hello\n')
        f.write('img2.jpg,world\n')
    
    with open(dummy_twitter_csv, 'w') as f:
        f.write('text,sentiment\n')
        f.write('This is a great movie,positive\n')
        f.write('I hate this,negative\n')
    
    iiit5k_data = load_iiit5k_data(dummy_iiit5k_csv, dummy_images_base_path)
    print(iiit5k_data[:2])
    
    twitter_data = load_twitter_sentiment_data(dummy_twitter_csv)
    print(twitter_data.head())
    
    # Los generadores requieren funciones de preprocesamiento reales
    # for batch_img, batch_lbl in image_data_generator(iiit5k_data, 1, lambda x: x):
    #     print(f"Batch de imagen: {batch_img}, Etiqueta: {batch_lbl}")
    #     break
    
    # for batch_txt, batch_sent in text_data_generator(twitter_data, 1, lambda x: x):
    #     print(f"Batch de texto: {batch_txt}, Sentimiento: {batch_sent}")
    #     break
