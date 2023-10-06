from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import pandas as pd
import sklearn

# Same coding step to using one data
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

y = df['label'].values



# Combine the feature matrices
x = np.hstack((X_text, X_title, X_author))
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Score and test:
test_acc = sklearn.model_selection.cross_val_score(knn, x_test, y_test, scoring = "accuracy")
print("The test accuracy is:", test_acc)
test_auc = sklearn.model_selection.cross_val_score(knn, x_test, y_test, scoring = "roc_auc")
print("The testing auc is:", test_auc)