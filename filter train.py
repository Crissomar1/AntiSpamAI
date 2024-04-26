import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import os
import mailbox

nltk.download('stopwords')  # Uncomment if stopwords haven't been downloaded
nltk.download('wordnet')  # Uncomment if WordNetLemmatizer isn't present

email_list = []
carpetas = ['easy_ham', 'spam']
for carpeta in carpetas:
    for archivo in os.listdir(carpeta):
        # Abrir el archivo como un buzón de correo
        mbox = mailbox.mbox(carpeta + '/' + archivo)
        # Agregar cada correo en el buzón a la lista de correos
        for correo in mbox:
            email_list.append(correo)
    #guardar la cantidad de correos en cada carpeta en ua variable
    if carpeta == 'easy_ham':
        easy_ham_count = len(email_list)
    else:
        spam_count = len(email_list) - easy_ham_count

print("Correos Cargados")

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
    encoded_email = np.zeros(len(vocab))

    # Asigna 1 a la posición correspondiente a cada palabra en el vocabulario
    for word in words:
        if word in vocab_dict:
            encoded_email[vocab_dict[word]] = 1

    return encoded_email


# Crea un vocabulario a partir de una lista de correos electrónicos
from collections import Counter

def create_vocabulary(email_list, frequency_threshold=5):
    # Initialize a Counter object to store word frequencies
    word_freq = Counter()

    # Iterate over each email in the list
    for email in email_list:
        # Convert the email into a list of preprocessed words
        words = preprocess_email(email).split()

        # Update word frequencies
        word_freq.update(words)

    # Filter out uncommon words
    vocab_list = [word for word, freq in word_freq.items() if freq >= frequency_threshold]

    # Sort the vocabulary list
    vocab_list.sort()

    return vocab_list

# # Crea el vocabulario a partir de los correos electrónicos
# vocab = create_vocabulary(email_list,3)

# # Guarda el vocabulario en un archivo de texto
# with open('vocab.txt', 'w') as f:
#     for word in vocab:
#         f.write(word + '\n')

# Carga el vocabulario desde el archivo de texto
with open('vocab.txt', 'r') as f:
    vocab = [line.strip() for line in f]

print("Vocabulario Cargado")

# cada correo electrónico en la lista de correos de entrenamiento y asigna una etiqueta dependiendo de la carpeta en la que se encuentre
# contar cuantos correos hay en cada carpeta usando listdir() y len()
spam_emails = email_list[:spam_count]
ham_emails = email_list[spam_count:]

# Agrega los correos electrónicos codificados a un DataFrame

# Crea un DataFrame para almacenar los correos electrónicos y sus etiquetas
email_df = pd.DataFrame(columns=['text', 'label'])

# Crea una lista de correos electrónicos codificados
encoded_emails = []
vocab_dict = {word: i for i, word in enumerate(vocab)}

# Codifica los correos electrónicos de spam
encoded_emails = [one_hot_encode(email, vocab_dict) for email in spam_emails]

print("Spam Codificado")

# Codifica los correos electrónicos de ham
encoded_emails += [one_hot_encode(email, vocab_dict) for email in ham_emails]

print("Ham Codificado")

# Agrega los correos electrónicos codificados al DataFrame

email_df['text'] = encoded_emails
email_df['label'] = ['spam'] * spam_count + ['ham'] * easy_ham_count

# Guarda el DataFrame en un archivo CSV
email_df.to_pickle('emails.pkl')

#imprime cuantos correos hay en cada carpeta
print(email_df['label'].value_counts())

#imprime el tamaño de el "texto" de un correo para verificar que se haya codificado correctamente
print(email_df['text'][0].shape)

