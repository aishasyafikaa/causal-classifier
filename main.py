
from UniCausal._datasets.unifiedcre import load_cre_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk import word_tokenize

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def extract_features(sentence: str):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)


    relators = ['because', 'since', 'after', 'as', 'cause', 'lead', 'make', 'result', 'bring']
    features = {}

    # Relator presence and position
    for word in relators:
        features[f"has_{word}"] = int(word in tokens)
        features[f"{word}_at_start"] = int(tokens[0] == word)

    # Length and punctuation
    features["length"] = len(tokens)
    features["punctuation_count"] = sum(1 for c in sentence if c in '.,;!?')

    return features

def main():
    print("Loading datasets: altlex, because, ctb, semeval2010t8")
    span_data, seqpair_data, stats = load_cre_dataset(
        dataset_name=['altlex', 'because', 'ctb', 'semeval2010t8'],
        do_train_val=True,
        data_dir='UniCausal/data'
    )
    #print("Available keys in seqpair_data:", seqpair_data.keys())

    #prepare training and validation data
    target_corpora = ['altlex', 'because', 'ctb', 'semeval2010t8']
    train_data = [ex for ex in seqpair_data['seq_train'] if ex['corpus'] in target_corpora]
    val_data = [ex for ex in seqpair_data['seq_validation'] if ex['corpus'] in target_corpora]

    #check how many examples available in datasets
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print("Train label distribution:", Counter([ex['label'] for ex in train_data]))
    print("Validation label distribution:", Counter([ex['label'] for ex in val_data]))

    print("Extracting features...")
    X_train = [extract_features(ex['text']) for ex in train_data]
    y_train = [ex['label'] for ex in train_data]

    X_test = [extract_features(ex['text']) for ex in val_data]
    y_test = [ex['label'] for ex in val_data]

    print("Vectorizing features...")
    vec = DictVectorizer(sparse=False)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    print("Training Decision Tree...")
    clf = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        criterion='entropy',
        random_state=42
    )
    clf.fit(X_train_vec, y_train)

    print("Evaluation:")
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    print("Visualizing decision tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=vec.get_feature_names_out(), class_names=["non-causal", "causal"], filled=True, rounded=True)
    plt.title("Decision Tree for Causal Sentence Classification (open-access datasets)")
    plt.savefig("decision_tree_visualization.png")
    plt.show()

if __name__ == "__main__":
    main()
