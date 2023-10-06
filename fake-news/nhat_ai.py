from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
df['author'].fillna('Unknown', inplace=True)
df['title'].fillna('Ambiguous', inplace=True)
df['text'].fillna('Ambiguous', inplace=True)
df.drop_duplicates(inplace=True)

# TF-IDF Vectorization for text and title
vectorizer_text = TfidfVectorizer(max_features=5000)
X_text = vectorizer_text.fit_transform(df['text']).toarray()

vectorizer_title = TfidfVectorizer(max_features=1000)
X_title = vectorizer_title.fit_transform(df['title']).toarray()

# One-hot encoding for authors
encoder = OneHotEncoder()
X_author = encoder.fit_transform(df[['author']]).toarray()

# Train-test split
y = df['label'].values
X_train_text, X_test_text, X_train_title, X_test_title, X_train_author, X_test_author, y_train, y_test = train_test_split(
    X_text, X_title, X_author, y, test_size=0.2, random_state=42)

# Neural Network Architecture
input_text = Input(shape=(X_train_text.shape[1],))
input_title = Input(shape=(X_train_title.shape[1],))
input_author = Input(shape=(X_train_author.shape[1],))

# Layers for text
x1 = Dense(128, activation='relu')(input_text)
x1 = Dense(64, activation='relu')(x1)

# Layers for title
x2 = Dense(128, activation='relu')(input_title)
x2 = Dense(64, activation='relu')(x2)

# Layers for author
x3 = Dense(128, activation='relu')(input_author)
x3 = Dense(64, activation='relu')(x3)

# Concatenate
concat = Concatenate()([x1, x2, x3])

# Final layers
out = Dense(64, activation='relu')(concat)
out = Dense(1, activation='sigmoid')(out)

# Compile model
model = Model(inputs=[input_text, input_title, input_author], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit([X_train_text, X_train_title, X_train_author], y_train, epochs=10, batch_size=32)

# Evaluate model
score = model.evaluate([X_test_text, X_test_title, X_test_author], y_test)
print(f"Test Accuracy: {score[1]}")

y_pred = model.predict([X_test_text, X_test_title, X_test_author])
y_pred = np.round(y_pred).flatten()  # Round the probabilities to get binary class labels

# Print classification report
print(classification_report(y_test, y_pred))
