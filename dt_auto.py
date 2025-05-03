import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
train_df = pd.read_csv('UniCausal/data/splits/because_train.csv')
test_df = pd.read_csv('UniCausal/data/splits/because_test.csv')

train_df['label'] = train_df['seq_label']
test_df['label'] = test_df['seq_label']

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

''''
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
grid_search = GridSearchCV(bagging, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)'''

'''#############################
# RANDOM FOREST
#############################
# Define RandomForest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
#############################'''

#############################
# Logistic Regression
#############################
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear', penalty='l2')
lr.fit(X_train_vec, y_train)
y_pred = lr.predict(X_test_vec)
###########################
'''grid_search.fit(X_train_vec, y_train)

# Print best params and score
print("Best Parameters:")
print(grid_search.best_params_)
print(f"Best F1 Score (CV): {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_vec)'''


print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))

''''# ========================
# Step 5: Try Your Own Sentences
# ========================

custom_sentences = [
    "Because it was raining, the game was canceled.",
    "He enjoys reading books at night.",
    "Due to the heavy traffic, I was late to work.",
    "The sun is bright today.",
    "she cried because he pushed her",
    "Good project leads to high mark."
]

# Lowercase to match training preprocessing
custom_sentences = [s.lower() for s in custom_sentences]

# Transform using fitted vectorizer
custom_vec = vectorizer.transform(custom_sentences)

# Predict using the best trained model from GridSearchCV
custom_preds = best_model.predict(custom_vec)

# Output predictions
for sent, pred in zip(custom_sentences, custom_preds):
    label = "Causal" if pred == 1 else "Non-Causal"
    print(f"\nSentence: {sent}\nPrediction: {label}")'''

