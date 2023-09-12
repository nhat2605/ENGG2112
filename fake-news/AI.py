import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv('train.csv')

# Step 2: Text Preprocessing
# You need to preprocess the text data using libraries like NLTK or SpaCy

# Step 3: Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X = tfidf_vectorizer.fit_transform(data['text'])

# Step 4: Data Splitting
X_train, X_val, y_train, y_val = train_test_split(X, data['label'], test_size=0.2, random_state=42)

# Step 5: Model Selection and Building
model = LogisticRegression()  # You can use other models too
model.fit(X_train, y_train)

# Step 6: Model Evaluation
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
classification_rep = classification_report(y_val, predictions)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)

# Step 9: Model Deployment (Saving the Model)
import joblib
joblib.dump(model, 'news_classifier_model.pkl')

# Step 10: Making Predictions
loaded_model = joblib.load('news_classifier_model.pkl')

# Assuming you have a new_news_article variable containing the preprocessed text
new_article_features = tfidf_vectorizer.transform([new_news_article])
prediction = loaded_model.predict(new_article_features)
if prediction[0] == 0:
    print('Fake News')
else:
    print('Reliable News')

# Step 11: Iterate and Improve
# You can iterate over different models, feature extraction methods, and hyperparameters to improve the model's performance.
