import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet as wn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

#nltk.download('wordnet')

# Load data
train_df = pd.read_csv('UniCausal/data/splits/esl2_train.csv')
test_df = pd.read_csv('UniCausal/data/splits/esl2_test.csv')
train_df['label'] = train_df['seq_label']
test_df['label'] = test_df['seq_label']

# Check counts in dataset
print("Train Examples: ", len(train_df))
print(train_df['label'].value_counts())

# Check counts in test set
print("Test Examples: ", len(test_df))
print(test_df['label'].value_counts())

# Deduplicate
train_df = train_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')
test_df = test_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')

# Combine and split
data = pd.concat([train_df, test_df], ignore_index=True)
data['label'] = data['seq_label']
X_train_text, X_test_text, y_train, y_test = train_test_split(
    data['text'], data['label'], stratify=data['label'], test_size=0.25, random_state=42
)

# Check label balance
print("Train label distribution:")
print(y_train.value_counts())
print("Test label distribution:")
print(y_test.value_counts())

# === Manual Feature Extraction ===
nlp = spacy.load('en_core_web_sm')
RELATORS = {'because', 'after', 'as', 'since'}
top_categories = ['causation', 'motion', 'state', 'change', 'response']

def get_verb_semantic_class(verb_lemma):
    synsets = wn.synsets(verb_lemma, pos=wn.VERB)
    return synsets[0].name() if synsets else 'none'

def map_synset_to_category(synset_name):
    for cat in top_categories:
        if cat in synset_name:
            return cat
    return 'other'

def extract_features(text):
    doc = nlp(text.lower())
    found_rels = set()
    found_mods = set()
    cause_class = 'none'
    effect_class = 'none'
    for i, token in enumerate(doc):
        if token.text in RELATORS:
            found_rels.add(token.text)
            if i > 0 and doc[i-1].pos_ in {"ADV", "ADP"}:
                found_mods.add("left_" + doc[i-1].pos_.lower())
            if i + 1 < len(doc) and doc[i+1].pos_ in {"ADP", "SCONJ"}:
                found_mods.add("right_" + doc[i+1].pos_.lower())
            for j in range(i-1, -1, -1):
                if doc[j].pos_ == "VERB":
                    cause_class = map_synset_to_category(get_verb_semantic_class(doc[j].lemma_))
                    break
            for j in range(i+1, len(doc)):
                if doc[j].pos_ == "VERB":
                    effect_class = map_synset_to_category(get_verb_semantic_class(doc[j].lemma_))
                    break
    return {
        'relator': ",".join(sorted(found_rels)) if found_rels else 'none',
        'relator_modifier': ",".join(sorted(found_mods)) if found_mods else 'none',
        'cause_verb_sem_class': cause_class,
        'effect_verb_sem_class': effect_class
    }

# Extract manual features
train_feats = pd.DataFrame(X_train_text.apply(extract_features).tolist())
test_feats = pd.DataFrame(X_test_text.apply(extract_features).tolist())

# One-hot encode
cat_cols = ['relator', 'relator_modifier', 'cause_verb_sem_class', 'effect_verb_sem_class']
train_feats = pd.get_dummies(train_feats, columns=cat_cols)
test_feats = pd.get_dummies(test_feats, columns=cat_cols)

# Align feature columns
missing_cols = set(train_feats.columns) - set(test_feats.columns)
for col in missing_cols:
    test_feats[col] = 0
test_feats = test_feats[train_feats.columns]

# === TF-IDF ===
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# === Combine Features ===
X_train_structured = train_feats.astype('float32').to_numpy()
X_test_structured = test_feats.astype('float32').to_numpy()

X_train_combined = hstack([X_train_tfidf, X_train_structured])
X_test_combined = hstack([X_test_tfidf, X_test_structured])

# === Train Classifier (Bagging + DT) ===
base_tree = DecisionTreeClassifier(random_state=42)
bagging = BaggingClassifier(estimator=base_tree, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5],
    'estimator__max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    bagging,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_combined, y_train)

# === Evaluation ===
print("Best Parameters:", grid_search.best_params_)
print(f"Best F1 Score (CV): {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_combined)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))

