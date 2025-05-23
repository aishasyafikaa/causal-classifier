import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("Loading train and test datasets...")

#testing all
from glob import glob
train_files = glob("UniCausal/data/splits/*_train.csv")
test_files = glob("UniCausal/data/splits/*_test.csv")

train_df = [pd.read_csv(file) for file in train_files]
train_df = pd.concat(train_df, ignore_index=True)

test_df = [pd.read_csv(file) for file in test_files]
test_df = pd.concat(test_df, ignore_index=True)

# Deduplicate
train_df = train_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')
test_df = test_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')

# Combine data
data = pd.concat([train_df, test_df], ignore_index=True)
data['label'] = data['seq_label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], stratify=data['label'], test_size=0.25, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


#############################
# DECISION TREE
#############################
# Define base model
base_tree = DecisionTreeClassifier(random_state=42)

# Define BaggingClassifier
bagging = BaggingClassifier(estimator=base_tree, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5],
    'estimator__max_features': ['sqrt', 'log2', None]
}

# Grid search with 5-fold CV
grid_search = GridSearchCV(bagging, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

grid_search.fit(X_train_vec, y_train)

# Print best params and score
print("Best Parameters:")
print(grid_search.best_params_)
print(f"Best F1 Score (CV): {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_vec)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))


