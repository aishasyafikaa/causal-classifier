import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score

# Load data
train_df = pd.read_csv('UniCausal/data/splits/because_train.csv')
test_df = pd.read_csv('UniCausal/data/splits/because_test.csv')

train_df['label'] = train_df['seq_label']
test_df['label'] = test_df['seq_label']

# Deduplicate based on corpus, doc_id, and sent_id — keep only the first occurrence
train_df = train_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')
test_df = test_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')

'''# extract feature and labels
x_train = train_df['text']
x_test = test_df['text']

y_train = train_df['label']
y_test = test_df['label']'''

# Check counts in dataset
print("Train Examples: ", len(train_df))
print(train_df['label'].value_counts())

# Check counts in test set
print("Test Examples: ", len(test_df))
print(test_df['label'].value_counts())

#for because data uneven split train test
# Combine train and test for stratified re-split
data = pd.concat([train_df, test_df], ignore_index=True)
data['label'] = data['seq_label']

# Stratified train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data['text'], data['label'], stratify=data['label'], test_size=0.25, random_state=42
)

# Check label balance
print("Train label distribution:")
print(y_train.value_counts())
print("Test label distribution:")
print(y_test.value_counts())

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Initial SVM training and evaluation
clf_svm = SVC(random_state=42, class_weight='balanced')
clf_svm.fit(x_train_tfidf, y_train)
y_pred = clf_svm.predict(x_test_tfidf)

print("Result of SVM:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))

'''ConfusionMatrixDisplay.from_estimator(clf_svm,
                                      x_test_tfidf,
                                      y_test,
                                      values_format='d',
                                      display_labels=["Non-Causal", "Causal"])
plt.title(f"SVM Confusion Matrix")
plt.show()'''

# Define kernels to test
kernels = ['linear', 'rbf']

# Benchmark each kernel
for kernel in kernels:
    print(f"\nEvaluating SVM with {kernel} kernel")

    # Define hyperparameter grid
    if kernel == 'linear':
        param_grid = {'C': [0.5, 1, 10, 100], 'kernel': ['linear']}
    else:  # RBF
        param_grid = {
            'C': [0.5, 1, 10, 100],
            'gamma': ['scale', 1, 0.1, 0.01, 0.001],
            'kernel': ['rbf']
        }
    # Grid search to optimize hyperparameters
    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='f1_macro', verbose=0)
    grid.fit(x_train_tfidf, y_train)

    # Get best model
    clf = grid.best_estimator_

    # Predict
    y_pred = clf.predict(x_test_tfidf)

    # Report
    print("Best Parameters:", grid.best_params_)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=2))

    '''ConfusionMatrixDisplay.from_estimator(clf,
                                          x_test_tfidf,
                                          y_test,
                                          values_format='d',
                                          display_labels=["Non-Causal", "Causal"])
    plt.title(f"SVM Confusion Matrix ({kernel} kernel)")
    plt.show()'''

# PCA for visualization
# ================================
# PCA for Visualisation
# ================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Apply PCA on TF-IDF features (convert to dense array first)
pca = PCA()
X_train_pca = pca.fit_transform(x_train_tfidf.toarray())

# Scree plot to show explained variance
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot (SemEval)')
plt.show()

# Use only PC1 and PC2 for decision surface
train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]
pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

# Define param grid (RBF kernel works better for curved boundaries)
param_grid = {
    'C': [0.5, 1, 10, 100],
    'gamma': ['scale', 0.1, 0.01],
    'kernel': ['rbf']
}

# Grid search on 2D PCA data for visualisation purposes only
grid_pca = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='accuracy', verbose=0)
grid_pca.fit(pca_train_scaled, y_train)
print("\nBest parameters for PCA-based SVM:", grid_pca.best_params_)

clf_svm_pca = grid_pca.best_estimator_
clf_svm_pca.fit(pca_train_scaled, y_train)

# Transform test set into PCA space
X_test_pca = pca.transform(x_test_tfidf.toarray())
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]
pca_test_scaled = scale(np.column_stack((test_pc1_coords, test_pc2_coords)))

# Generate mesh for decision surface
x_min, x_max = pca_test_scaled[:, 0].min() - 1, pca_test_scaled[:, 0].max() + 1
y_min, y_max = pca_test_scaled[:, 1].min() - 1, pca_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf_svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision surface
fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(xx, yy, Z, alpha=0.1)

# Plot actual test points
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])  # red = non-causal, green = causal
scatter = ax.scatter(pca_test_scaled[:, 0], pca_test_scaled[:, 1], c=y_test,
                     cmap=cmap, s=100, edgecolors='k', alpha=0.7)

# Add legend
legend = ax.legend(*scatter.legend_elements(), loc="upper right")
legend.get_texts()[0].set_text("Non-Causal")
legend.get_texts()[1].set_text("Causal")

# Labels and title
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("SVM Decision Surface (PCA-transformed TF-IDF, Because)")
plt.show()
