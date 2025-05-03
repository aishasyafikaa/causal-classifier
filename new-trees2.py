from UniCausal._datasets.unifiedcre import load_cre_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_features(text: str):
    text_lower = text.lower()
    doc = nlp(text_lower)

    # Causal keywords and bigrams
    relators = ['because', 'since', 'after', 'as', 'cause', 'lead', 'make', 'result', 'bring']
    bigrams = ['due to', 'so that', 'leads to', 'as a result', 'results in']

    features = {}

    # Relators
    tokens = [token.text for token in doc]
    for rel in relators:
        features[f"has_{rel}"] = int(rel in tokens)
        features[f"{rel}_at_start"] = int(tokens[0] == rel)

    features['causal_keyword_count'] = sum(features[f"has_{rel}"] for rel in relators)

    # Bigrams
    text_joined = " ".join(tokens)
    for phrase in bigrams:
        features[f"has_bigram_{phrase.replace(' ', '_')}"] = int(phrase in text_joined)

    # POS tags
    features['num_verbs'] = sum(1 for token in doc if token.pos_ == 'VERB')
    features['num_nouns'] = sum(1 for token in doc if token.pos_ == 'NOUN')
    features['num_adjs'] = sum(1 for token in doc if token.pos_ == 'ADJ')

    # Dependency labels
    features['num_advcl'] = sum(1 for token in doc if token.dep_ == 'advcl')
    features['num_mark'] = sum(1 for token in doc if token.dep_ == 'mark')

    # ARG tags
    features['has_ARG0'] = int('<ARG0>' in text)
    features['has_ARG1'] = int('<ARG1>' in text)

    # Length and punctuation
    features["length"] = len(tokens)
    features["punctuation_count"] = sum(1 for c in text if c in '.,;!?')

    return features

def main():
    print("Loading datasets...")
    span_data, seqpair_data, stats = load_cre_dataset(
        dataset_name=['altlex'],
        do_train_val=True,
        data_dir='UniCausal/data'
    )

    train_data = seqpair_data['seq_train']
    val_data = seqpair_data['seq_validation']

    print(f"📚 Train examples: {len(train_data)}")
    print(f"🧪 Validation examples: {len(val_data)}")
    print("🔢 Train label distribution:", Counter([ex['label'] for ex in train_data]))
    print("🔢 Validation label distribution:", Counter([ex['label'] for ex in val_data]))

    print("🧠 Extracting features...")
    X_train = [extract_features(ex['text']) for ex in train_data]
    y_train = [ex['label'] for ex in train_data]

    X_test = [extract_features(ex['text']) for ex in val_data]
    y_test = [ex['label'] for ex in val_data]

    print("📊 Vectorizing...")
    vec = DictVectorizer(sparse=False)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    print("🌲 Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train_vec, y_train)

    print("📈 Evaluation:")
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    print("❌ Misclassified examples:")
    for ex, true, pred in zip(val_data, y_test, y_pred):
        if true != pred:
            print(f"- Text: {ex['text']}\n  True: {true}, Predicted: {pred}\n")

    print("🌟 Top 10 most important features:")
    importances = clf.feature_importances_
    feature_names = vec.get_feature_names_out()
    top_features = sorted(zip(importances, feature_names), reverse=True)[:10]
    for score, name in top_features:
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()


