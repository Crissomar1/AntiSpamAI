import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import os
import mailbox
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')

def flatten_payload(email):
    email_text = email.get_payload()
    if isinstance(email_text, list):
        return ' '.join(flatten_payload(part) for part in email_text)
    else:
        return email_text

def preprocess_email(email):
    email_text = email.get_payload()
    email_text = flatten_payload(email)
    # 1. Conversión a minúsculas:
    email_text = email_text.lower()

    # 2. Eliminación de caracteres especiales y URLs:
    email_text = re.sub(r'[^a-z0-9\s]', ' ', email_text)  # Sustituye caracteres especiales
    email_text = re.sub(r'http\S+', ' ', email_text)  # Elimina URLs

    # 3. Tokenización (separa el texto en palabras):
    words = email_text.split()

    # 4. Eliminación de stopwords (palabras de enlace frecuentes):
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 5. Lematización (reduce palabras a su raíz):
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def one_hot_encode(text, vocab_dict):
    # Convierte el texto en una lista de palabras preprocesadas
    words = preprocess_email(text)

    # Crea un vector de ceros del tamaño del vocabulario
    encoded_email = np.zeros(len(vocab_dict))

    # Asigna 1 a la posición correspondiente a cada palabra en el vocabulario
    for word in words:
        if word in vocab_dict:
            encoded_email[vocab_dict[word]] += 1

    return encoded_email

def read_vocab() -> list:
    if os.path.exists('vocab.txt'):
        with open('vocab.txt', 'r') as f:
            vocab = [line.strip() for line in f]
        vocab_dict = {word: i for i, word in enumerate(vocab)}
    else:
        vocab_dict = {}
    return vocab_dict

def encode_email(dir, vocab_dict):
    # Carga el correo electrónico
    email = [email for email in mailbox.mbox(dir)][0]
    encoded_email = one_hot_encode(email, vocab_dict)
    return encoded_email
