import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

#===========================
# load and prepare dataset
#===========================
print("Loading train and test datasets...")
train_df = pd.read_csv('UniCausal/data/splits/esl2_train.csv')
test_df = pd.read_csv('UniCausal/data/splits/esl2_test.csv')

#This codes only apply when using text only, not text with pairs
#train_df = train_df[train_df["eg_id"] == 0]
#test_df = test_df[test_df["eg_id"] == 0]

train_df['label'] = train_df['pair_label']
test_df['label'] = test_df['pair_label']

train_texts = train_df['text_w_pairs'].tolist()
test_texts = test_df['text_w_pairs'].tolist()

train_labels = train_df['label'].tolist()
test_labels = test_df['label'].tolist()


#=================================
# Tokenize
#=================================
print("Tokenizing...")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128)
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=128)

# ========================
# Step 3: Wrap in Dataset
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
# Step 4: Load Model
# ========================
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ========================
# Step 5: Train with Trainer
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
# Step 6: Train!
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


