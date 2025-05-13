import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

#===========================
# load and prepare dataset
#===========================
print("Loading train and test datasets...")

#testing all
from glob import glob
train_files = glob("UniCausal/data/splits/*_train.csv")
test_files = glob("UniCausal/data/splits/*_test.csv")

train_df = [pd.read_csv(file) for file in train_files]
train_df = pd.concat(train_df, ignore_index=True)

test_df = [pd.read_csv(file) for file in test_files]
test_df = pd.concat(test_df, ignore_index=True)


'''train_df = pd.read_csv('UniCausal/data/splits/because_train.csv')
test_df = pd.read_csv('UniCausal/data/splits/because_test.csv')

print(f"Train set: {len(train_df)} rows")
print(f"Test set: {len(test_df)} rows")'''

# Deduplicate
train_df = train_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')
test_df = test_df.sort_values('eg_id').drop_duplicates(subset=['corpus', 'doc_id', 'sent_id'], keep='first')

combined_df = pd.concat([train_df, test_df], ignore_index=True)

combined_df['label'] = combined_df['seq_label']
texts = combined_df['text'].tolist()
labels = combined_df['label'].astype(int).tolist()

print("Combined label distribution:")
print(combined_df['label'].value_counts())

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, stratify=labels, test_size=0.25, random_state=42
)

from collections import Counter

print("Train label distribution:")
print(Counter(train_labels))

print("Test label distribution:")
print(Counter(test_labels))


#=================================
# Tokenize
#=================================
print("Tokenizing...")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128)
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=128)

# ========================
# Wrap in Dataset
# ========================
class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CausalDataset(train_encodings, train_labels)
test_dataset = CausalDataset(test_encodings, test_labels)

# ========================
# Load Model
# ========================
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ========================
# Train with Trainer
# ========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    seed = 42,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # using the test set for validation
)

# ========================
# Step 6: Train
# ========================
print("Training the model...")
trainer.train()

# ========================
# Step 7: Evaluate
# ========================
print("Evaluation on test set:")
trainer.evaluate()

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Get model predictions
print("Generating predictions...")
pred_output = trainer.predict(test_dataset)

# Step 2: Convert logits to predicted class labels
pred_logits = pred_output.predictions
pred_labels = np.argmax(pred_logits, axis=1)

# Step 3: Print classification report
print("Classification Report:")
print(classification_report(test_labels, pred_labels, digits=2))

# Step 4: Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(test_labels, pred_labels))

for i, (text, pred, label) in enumerate(zip(test_texts, pred_labels, test_labels)):
    if pred != label:
        print(f"[WRONG] Text: {text}\n→ Predicted: {pred}, True: {label}\n")


