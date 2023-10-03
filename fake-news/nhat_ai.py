import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("train.csv")
df['author'].fillna('Unknown', inplace=True)
df['title'].fillna('Ambiguous', inplace=True)
df['text'].fillna('Ambiguous', inplace=True)
df.drop_duplicates(inplace=True)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
test = df['text']
X_text = vectorizer.fit_transform(test)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier())
])

# Hyperparameter grid
param_grid = {
    'tfidf__max_features': [2000, 5000, None],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20, 30],
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
