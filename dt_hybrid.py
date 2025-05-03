import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet as wn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import numpy as np
from sklearn.model_selection import train_test_split

# Download WordNet
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# ========================
# Step 1: Load Data
# ========================
# Load data
train_df = pd.read_csv('UniCausal/data/splits/altlex_train.csv')
test_df = pd.read_csv('UniCausal/data/splits/altlex_test.csv')
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

# ========================
# Step 2: Manual Feature Extraction
# ========================
RELATORS = {'because', 'after', 'as', 'since'}
top_categories = ['causation', 'motion', 'state', 'change', 'response']

def get_verb_semantic_class(verb_lemma):
    synsets = wn.synsets(verb_lemma, pos=wn.VERB)
    if synsets:
        return synsets[0].name()
    return 'none'

def map_synset_to_category(synset_name):
    for category in top_categories:
        if category in synset_name:
            return category
    return 'other'

def extract_features(text):
    doc = nlp(text.lower())
    found_rels = set()
    found_mods = set()
    cause_verb_sem_class = 'none'
    effect_verb_sem_class = 'none'

    for i, token in enumerate(doc):
        if token.text in RELATORS:
            found_rels.add(token.text)

            if i > 0 and doc[i - 1].pos_ in {"ADV", "ADP"}:
                found_mods.add('left_' + doc[i - 1].pos_.lower())
            if i + 1 < len(doc) and doc[i + 1].pos_ in {"ADP", "SCONJ"}:
                found_mods.add('right_' + doc[i + 1].pos_.lower())

            for j in range(i - 1, -1, -1):
                if doc[j].pos_ == "VERB":
                    lemma = doc[j].lemma_
                    synset_name = get_verb_semantic_class(lemma)
                    cause_verb_sem_class = map_synset_to_category(synset_name)
                    break
            for j in range(i + 1, len(doc)):
                if doc[j].pos_ == "VERB":
                    lemma = doc[j].lemma_
                    synset_name = get_verb_semantic_class(lemma)
                    effect_verb_sem_class = map_synset_to_category(synset_name)
                    break

    relator = ",".join(sorted(found_rels)) if found_rels else 'none'
    modifier_type = ",".join(sorted(found_mods)) if found_mods else 'none'

    return {
        'relator': relator,
        'relator_modifier': modifier_type,
        'cause_verb_sem_class': cause_verb_sem_class,
        'effect_verb_sem_class': effect_verb_sem_class
    }

# Apply feature extraction
train_feats = X_train_text.apply(extract_features)
test_feats = X_test_text.apply(extract_features)

train_feats_df = pd.DataFrame(train_feats.tolist())
test_feats_df = pd.DataFrame(test_feats.tolist())

# Add labels
train_feats_df['label'] = y_train.values
test_feats_df['label'] = y_test.values

# ========================
# Step 3: Prepare Manual Features
# ========================
train_feats_df = pd.get_dummies(train_feats_df, columns=[
    'relator', 'relator_modifier', 'cause_verb_sem_class', 'effect_verb_sem_class'])
test_feats_df = pd.get_dummies(test_feats_df, columns=[
    'relator', 'relator_modifier', 'cause_verb_sem_class', 'effect_verb_sem_class'])

missing_cols = set(train_feats_df.columns) - set(test_feats_df.columns)
for col in missing_cols:
    test_feats_df[col] = 0
test_feats_df = test_feats_df[train_feats_df.columns]

X_train_manual = train_feats_df.drop(columns=['label']).astype('float64')
X_test_manual = test_feats_df.drop(columns=['label']).astype('float64')
y_train = train_feats_df['label']
y_test = test_feats_df['label']

# ========================
# Step 4: TF-IDF
# ========================
X_train_lower = X_train_text.str.lower()
X_test_lower = X_test_text.str.lower()

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_lower)
X_test_tfidf = vectorizer.transform(X_test_lower)


# ========================
# Step 5: Combine Features
# ========================
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_manual.values)])
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_manual.values)])

# ========================
# Step 6: Train RandomForest + Evaluate
# ========================
clf = RandomForestClassifier(
    n_estimators=50,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train_combined, y_train)
y_pred = clf.predict(X_test_combined)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))

# ========================
# Step 7: Feature Importance
# ========================
tfidf_feature_names = vectorizer.get_feature_names_out()
manual_feature_names = X_train_manual.columns
all_feature_names = np.concatenate([tfidf_feature_names, manual_feature_names])

importances = clf.feature_importances_
feature_importance_pairs = list(zip(all_feature_names, importances))
sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

print("\nTop 20 Important Features:")
for feature, importance in sorted_features[:20]:
    print(f"{feature:<35} {importance:.5f}")
