# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:00:11 2024

@author: pradh
"""
import torch
from transformers import Trainer
from transformers import BertTokenizer, BertForSequenceClassification

import numpy as np
import pandas as pd

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
    
model_name = 'google-bert/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
       
# Load test data
test_path = r"D:\kaggle\LLM_Detect_AI_Generated_Text\Data\test_sub.csv"
test_data = pd.read_csv(test_path)
X_test = list(test_data["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = r"D:\kaggle\LLM_Detect_AI_Generated_Text\trained_model"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

###Metrics
from sklearn import metrics

actual = test_data['label']
predicted = y_pred

confusion_matrix = metrics.confusion_matrix(actual, predicted)

Accuracy = metrics.accuracy_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
Precision = metrics.precision_score(actual, predicted)
F1_score = metrics.f1_score(actual, predicted)
Sensitivity_recall = metrics.recall_score(actual, predicted)

#metrics
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})
