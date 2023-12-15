# -*- coding: utf-8 -*-
from dataset import dataset
from datasets import Dataset # converting df to transformer dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import numpy as np
import os
import torch
import warnings
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_id = 0 if str(device) == 'cuda' else -1

FNC_PATH="./dataset/FNC-1"
LIAR_PATH="./dataset/LIAR/"

ds = dataset(FNC_PATH=FNC_PATH, LIAR_PATH=LIAR_PATH, word2vec=False)
train_df, val_df, test_df = ds(dataset="LIAR", all = True)

bert_model = pipeline('fill-mask', model='bert-base-cased', device=device_id)

# utils
def get_prediction(bert_respond)->int:
    return 1 if sorted(bert_respond, key = lambda x: x['score'])[-1]['token_str'] == "real" else 0

train_X = train_df['statement']
train_y = train_df['label'].astype(int).to_numpy()

val_X = val_df['statement']
val_y = val_df['label'].astype(int).to_numpy()

test_X = test_df['statement']
test_y = test_df['label'].astype(int).to_numpy()

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def filter_long_sample(X_df, y_df, print_head = ""):
    tokenized_X = X_df.apply(lambda x: tokenizer(x))
    too_long    = tokenized_X.apply(lambda x: len(x['input_ids']) > 512)
    print(f"{print_head} filtered {sum(too_long)} sample(s), remain {sum(~too_long)} samples.")
    return X_df[~too_long], y_df[~too_long]

train_X, train_y = filter_long_sample(train_X, train_y, print_head="Train dataset: ")
val_X, val_y     = filter_long_sample(val_X, val_y, print_head="Val dataset: ")
test_X, test_y   = filter_long_sample(test_X, test_y, print_head="Test dataset: ")

# basic bert without training
result = test_X.apply(lambda x: bert_model(f"{x} This is a [MASK] news.", targets=["real", "fake"]) )
prediction = result.apply(lambda x: get_prediction(x) )

print(f"Basic Bert Result: ")
print(f"\tAccuracy: {accuracy_score(test_y, prediction.values):.4f}")
print(f"\tF1 score: {f1_score(test_y, prediction.values):.4f}")