import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

# Read the vocab.txt file
with open('vocab.txt', 'r') as file:
    vocab = file.read().splitlines()

# Read the emails.pkl dataframe
emails_df = pd.read_pickle('emails.pkl')

# Shuffle the dataframe
emails_df = emails_df.sample(frac=1).reset_index(drop=True)

# Separate the data into train, test, and evaluation sets
train_df, test_df, eval_df = train_test_split(emails_df, test_size=0.2, random_state=42)
test_df, eval_df = train_test_split(test_df, test_size=0.5, random_state=42)

# Create he anti-spam AI model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(len(vocab),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])