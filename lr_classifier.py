import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load and prepare data
train_df = pd.read_csv('UniCausal/data/splits/semeval2010t8_train.csv')
test_df = pd.read_csv('UniCausal/data/splits/semeval2010t8_test.csv')

# Use 'pair_label' consistently for AltLex (pair classification)
train_df['label'] = train_df['pair_label']
test_df['label'] = test_df['pair_label']

'''# Deduplicate (only for seq task)
train_df = train_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')
test_df = test_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')'''

# Combine data
data = pd.concat([train_df, test_df], ignore_index=True)

# Split into train/test again (optional: you could use original split)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], stratify=data['label'], test_size=0.25, random_state=42
)

# TF-IDF vectorisation
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='liblinear',
    penalty='l2'
)

lr.fit(X_train_vec, y_train)
y_pred = lr.predict(X_test_vec)

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))
