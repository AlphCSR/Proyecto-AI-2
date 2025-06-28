
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos de NLTK si no están presentes
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text_for_sentiment(text):
    """
    Limpia y preprocesa un texto para el análisis de sentimiento.
    """
    text = text.lower() # Convertir a minúsculas
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Eliminar URLs
    text = re.sub(r'@\w+', '', text) # Eliminar menciones
    text = re.sub(r'#\w+', '', text) # Eliminar hashtags
    text = re.sub(r'[^a-z\s]', '', text) # Eliminar caracteres no alfabéticos
    
    tokens = text.split() # Tokenizar
    tokens = [word for word in tokens if word not in stop_words] # Eliminar stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lematización
    
    return ' '.join(tokens)

def build_sentiment_model(vocab_size, embedding_dim, max_sequence_length, num_classes):
    """
    Construye un modelo RNN (LSTM) para clasificación de sentimiento.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # 3 clases: positivo, negativo, neutral
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_sentiment_model(model, texts, sentiments, epochs=10, batch_size=32):
    """
    Entrena el modelo de sentimiento.
    texts: Lista de textos preprocesados.
    sentiments: Lista de etiquetas de sentimiento (ej. [0, 1, 2] para neg, neu, pos).
    """
    # Tokenización y padding
    tokenizer = Tokenizer(num_words=len(set(texts)), oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1])
    
    # Convertir sentimientos a one-hot encoding
    num_classes = model.output_shape[1]
    sentiment_labels = tf.keras.utils.to_categorical(sentiments, num_classes=num_classes)
    
    model.fit(padded_sequences, sentiment_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model, tokenizer

def predict_sentiment(model, tokenizer, text, max_sequence_length):
    """
    Predice el sentimiento de un texto dado.
    """
    processed_text = preprocess_text_for_sentiment(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    
    prediction = model.predict(padded_sequence)[0]
    # Asumiendo que las clases son 0: negativo, 1: neutral, 2: positivo
    sentiment_map = {0: 'negativo', 1: 'neutral', 2: 'positivo'}
    predicted_class = np.argmax(prediction)
    return sentiment_map[predicted_class], prediction

if __name__ == '__main__':
    print("Ejecutando ejemplos de sentiment_model.py")
    
    # Ejemplo de preprocesamiento
    sample_text = "This is a great movie! Check it out: https://example.com @user #awesome"
    cleaned_text = preprocess_text_for_sentiment(sample_text)
    print(f"Texto original: {sample_text}")
    print(f"Texto preprocesado: {cleaned_text}")
    
    # Ejemplo de construcción de modelo
    vocab_size = 10000 # Tamaño del vocabulario
    embedding_dim = 100
    max_sequence_length = 100 # Longitud máxima de las secuencias
    num_classes = 3 # Negativo, Neutral, Positivo
    
    sentiment_model = build_sentiment_model(vocab_size, embedding_dim, max_sequence_length, num_classes)
    sentiment_model.summary()
    
    # Ejemplo de entrenamiento (datos dummy)
    dummy_texts = [
        "this movie is amazing",
        "i hate this product",
        "it was okay, nothing special",
        "best day ever",
        "terrible experience",
        "just another day"
    ]
    dummy_sentiments = [2, 0, 1, 2, 0, 1] # 0:neg, 1:neu, 2:pos
    
    # Para el tokenizer, necesitamos un conjunto de datos más grande o pre-entrenado
    # Aquí, solo para demostración, usaremos un tokenizer simple
    dummy_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    dummy_tokenizer.fit_on_texts(dummy_texts)
    
    # Entrenar el modelo (esto es solo un ejemplo, no un entrenamiento real)
    # sentiment_model, trained_tokenizer = train_sentiment_model(sentiment_model, dummy_texts, dummy_sentiments)
    
    # Ejemplo de predicción
    # test_text = "this is a fantastic film"
    # predicted_sentiment, probabilities = predict_sentiment(sentiment_model, trained_tokenizer, test_text, max_sequence_length)
    # print(f"Texto: '{test_text}' -> Sentimiento: {predicted_sentiment} (Probabilidades: {probabilities})")
