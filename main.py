import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


def load_data(train_path, test_path, label_type='pair_label', deduplicate=False):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['label'] = train_df[label_type]
    test_df['label'] = test_df[label_type]

    if deduplicate:
        train_df = train_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')
        test_df = test_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')

    data = pd.concat([train_df, test_df], ignore_index=True)
    return data


def prepare_data(data, ngram_range=(1, 3)):
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], stratify=data['label'], test_size=0.25, random_state=42
    )

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))


def run_logistic_regression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, class_weight='balanced',
        solver='liblinear', penalty='l2'
    )
    lr.fit(X_train, y_train)
    print("Logistic Regression Results:")
    evaluate_model(lr, X_test, y_test)


def run_svm(X_train, X_test, y_train, y_test):
    param_grid = {
        'C': [0.5, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='f1_macro', verbose=0)
    grid.fit(X_train, y_train)
    print("SVM Results:")
    print("Best Parameters:", grid.best_params_)
    evaluate_model(grid.best_estimator_, X_test, y_test)


def run_decision_tree_bagging(X_train, X_test, y_train, y_test):
    base_tree = DecisionTreeClassifier(random_state=42)
    bagging = BaggingClassifier(estimator=base_tree, random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'estimator__max_depth': [None, 10],
        'estimator__min_samples_split': [2, 5],
        'estimator__max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(bagging, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Decision Tree + Bagging Results:")
    print("Best Parameters:", grid_search.best_params_)
    evaluate_model(grid_search.best_estimator_, X_test, y_test)

def run_random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier

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

    grid_search.fit(X_train, y_train)

    print("Random Forest Results:")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Best F1 Score (CV): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_test, y_test)



# =========================
# ===== Main Runner =======
# =========================

if __name__ == '__main__':
    # Change these to the dataset you want to use
    train_path = 'UniCausal/data/splits/altlex_train.csv'
    test_path = 'UniCausal/data/splits/altlex_test.csv'
    label_type = 'seq_label'  # or 'seq_label'
    deduplicate = True

    data = load_data(train_path, test_path, label_type, deduplicate)
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Call the model(s) you want to run:
    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_svm(X_train, X_test, y_train, y_test)
    run_decision_tree_bagging(X_train, X_test, y_train, y_test)
    run_random_forest(X_train, X_test, y_train, y_test)
