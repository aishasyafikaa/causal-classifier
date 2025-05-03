import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet as wn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# nltk.download('wordnet')

# ========================
# Step 1: Load Preprocessed Data
# ========================
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

# ========================
# Step 2: Feature Extraction (Manual Linguistic Features)
# ========================
nlp = spacy.load('en_core_web_sm')
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
    cause_verb_class = 'none'
    effect_verb_class = 'none'

    for i, token in enumerate(doc):
        if token.text in RELATORS:
            found_rels.add(token.text)

            if i > 0 and doc[i - 1].pos_ in {"ADV", "ADP"}:
                found_mods.add('left_' + doc[i - 1].pos_.lower())
            if i + 1 < len(doc) and doc[i + 1].pos_ in {"ADP", "SCONJ"}:
                found_mods.add('right_' + doc[i + 1].pos_.lower())

            for j in range(i - 1, -1, -1):
                if doc[j].pos_ == "VERB":
                    cause_verb_class = map_synset_to_category(get_verb_semantic_class(doc[j].lemma_))
                    break
            for j in range(i + 1, len(doc)):
                if doc[j].pos_ == "VERB":
                    effect_verb_class = map_synset_to_category(get_verb_semantic_class(doc[j].lemma_))
                    break

    return {
        'relator': ",".join(sorted(found_rels)) if found_rels else 'none',
        'relator_modifier': ",".join(sorted(found_mods)) if found_mods else 'none',
        'cause_verb_sem_class': cause_verb_class,
        'effect_verb_sem_class': effect_verb_class
    }

# Apply feature extraction
train_feats = pd.DataFrame(X_train_text.apply(extract_features).tolist())
test_feats = pd.DataFrame(X_test_text.apply(extract_features).tolist())

# Add labels
train_feats['label'] = y_train.values
test_feats['label'] = y_test.values

# ========================
# Step 3: Feature Matrix Preparation
# ========================
# One-hot encode categorical features
cat_cols = ['relator', 'relator_modifier', 'cause_verb_sem_class', 'effect_verb_sem_class']
train_feats = pd.get_dummies(train_feats, columns=cat_cols)
test_feats = pd.get_dummies(test_feats, columns=cat_cols)

# Align columns
missing_cols = set(train_feats.columns) - set(test_feats.columns)
for col in missing_cols:
    test_feats[col] = 0
test_feats = test_feats[train_feats.columns]

# Final train/test matrix
X_train_vec = train_feats.drop(columns=['label']).astype('float32')
y_train = train_feats['label']
X_test_vec = test_feats.drop(columns=['label']).astype('float32')
y_test = test_feats['label']

# ========================
# Step 4: Bagging + Decision Tree + Grid Search
# ========================
base_tree = DecisionTreeClassifier(random_state=42)

bagging = BaggingClassifier(estimator=base_tree, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5],
    'estimator__max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    estimator=bagging,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_vec, y_train)

'''# ========================
# Step 4: RandomForestt + Grid Search
# ========================

rf = RandomForestClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_vec, y_train)'''

# ========================
# Step 5: Evaluate Best Model
# ========================
print("\nBest Parameters:")
print(grid_search.best_params_)
print(f"Best F1 Score (CV): {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_vec)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))

''''# ========================
# Step 6: Show Misclassified Sentences
# ========================

# Compare predictions with actual labels
misclassified_mask = y_pred != y_test.values
misclassified_df = test_df[misclassified_mask].copy()
misclassified_df['predicted_label'] = y_pred[misclassified_mask]

# Print misclassified examples
print("\nMisclassified Sentences:")
for i, row in misclassified_df.iterrows():
    print(f"\nText: {row['text']}\nTrue Label: {row['label']}\nPredicted: {row['predicted_label']}")'''
