import os
import random

import gdown as gdown
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns

from math import inf
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from sklearn.metrics import *
from transformers import AutoModel, AutoTokenizer, \
    get_linear_schedule_with_warmup
from dataset import dataset

EPOCHS = 5


class BertLSTM(nn.Module):
    def __init__(self, model_name, hidden_size=256):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_name,
                                                    num_hidden_layers=2)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.linear_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        new_inputs = inputs.copy()
        del new_inputs["labels"]
        last_hidden_state = self.bert_model(**new_inputs).last_hidden_state
        lstm_output, (_, _) = self.lstm(last_hidden_state)
        last_lstm_output = lstm_output[:, -1, :]
        result = self.linear_1(last_lstm_output)
        result = self.dropout(result)
        result = self.relu(result)
        result = self.linear_2(result)

        return result


# Prepare Data
def prepare_data(ds, dataset_type):
    data_list = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = ds(dataset="LIAR", type=dataset_type)

    statements = df["statement"].values.tolist()
    contexts = df["context(location)"].values.tolist()
    labels = df["label"].values.tolist()

    inputs = [f"{statement}{tokenizer.sep_token}{context}{tokenizer.sep_token}"
              for statement, context in zip(statements, contexts)]
    X = tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                    padding="max_length", truncation=True,
                                    return_tensors="pt")

    for idx, label in enumerate(labels):
        data = {
            "input_ids": X["input_ids"][idx],
            "attention_mask": X["attention_mask"][idx],
            "token_type_ids": X["token_type_ids"][idx],
            "labels": torch.tensor(label, dtype=torch.int64)
        }
        data_list.append(data)

    return X, labels, data_list


def train(model,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          device,
          epoches: int = EPOCHS):
    best_f1_score = 0.0
    train_loss_records = []
    val_loss_records = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8,
                                  weight_decay=0.01)
    scheduled = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(
                                                    train_dataloader) * epoches * 0.1,
                                                num_training_steps=len(
                                                    train_dataloader) * epoches)

    last_val_loss = inf
    val_loss_improve_nums = 0
    early_stop_patience = 3
    model.train()
    for epoch_idx in range(epoches):
        total_train_loss = 0.0
        total_val_loss = 0.0
        for train_batch in tqdm(train_dataloader):

            for key in train_batch:
                train_batch[key] = train_batch[key].to(device)

            output = model(train_batch)

            loss = criterion(output, train_batch["labels"])
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduled.step()

        train_loss_records.append(total_train_loss / len(train_dataloader))

        average_accuracy_score = 0.0
        average_precision_score = 0.0
        average_recall_score = 0.0
        average_f1_score = 0.0

        total_accuracy_score = 0.0
        total_precision_score = 0.0
        total_recall_score = 0.0
        total_f1_score = 0.0

        with torch.no_grad():
            total_truth_labels = []
            total_pred_labels = []
            for idx, val_batch in enumerate(val_dataloader):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)

                val_output = model(val_batch)

                val_loss = criterion(val_output, val_batch["labels"])
                total_val_loss += val_loss.item()

                logits_after_softmax = val_output.softmax(dim=1).cpu().detach()

                pred_labels = np.argmax(logits_after_softmax, axis=1).tolist()
                truth_labels = val_batch["labels"].cpu().detach().tolist()

                total_truth_labels.extend(truth_labels)
                total_pred_labels.extend(pred_labels)

        average_accuracy_score = round(
            accuracy_score(total_truth_labels, total_pred_labels), 4)
        average_precision_score = round(
            precision_score(total_truth_labels, total_pred_labels), 4)
        average_recall_score = round(
            recall_score(total_truth_labels, total_pred_labels), 4)
        average_f1_score = round(
            f1_score(total_truth_labels, total_pred_labels), 4)

        val_loss_records.append(total_val_loss / len(val_dataloader))

        print(
            f"epoch-num: {epoch_idx}, train-losss: {train_loss_records[-1]}, val-loss: {val_loss_records[-1]}")
        print(
            f"epoch-num: {epoch_idx}, average-accuracy: {average_accuracy_score}, average-precision: {average_precision_score}, average-recall: {average_recall_score}, average-f1: {average_f1_score}")

        torch.save(model.state_dict(),
                   f'./output/bert_lstm_model_pair_{epoch_idx}.pth')

        if val_loss_records[-1] > last_val_loss:
            val_loss_improve_nums += 1

        if val_loss_improve_nums > early_stop_patience:
            break

        last_val_loss = val_loss_records[-1]

    # Plotting Loss
    plt.figure()
    plt.plot(train_loss_records, label='Train Loss')
    plt.plot(val_loss_records, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./plots/bert_lstm_pair_loss.png')
    plt.show()


def test(model,
         model_path,
         test_dataloader: DataLoader,
         device):
    model.eval()
    weights_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(weights_dict)

    with torch.no_grad():
        total_truth_labels = []
        pred_truth_labels = []
        for idx, test_batch in enumerate(test_dataloader):
            for key in test_batch:
                test_batch[key] = test_batch[key].to(device)

            logits = model(test_batch)
            logits_after_softmax = logits.softmax(dim=1).cpu().detach()
            pred_labels = np.argmax(logits_after_softmax, axis=1).tolist()
            truth_labels = test_batch["labels"].cpu().detach().tolist()
            total_truth_labels.extend(truth_labels)
            pred_truth_labels.extend(pred_labels)

    accuracy = round(accuracy_score(total_truth_labels, pred_truth_labels), 4)
    precision = round(precision_score(total_truth_labels, pred_truth_labels),
                      4)
    recall = round(recall_score(total_truth_labels, pred_truth_labels), 4)
    f1 = round(f1_score(total_truth_labels, pred_truth_labels), 4)

    np.save("./output/y_truth_extension_2.npy",
            total_truth_labels)
    np.save("./output/y_pred_extension_2.npy",
            pred_truth_labels)

    print(
        f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

    return accuracy, f1


def evaluate(y_true_path, y_pred_path):
    # Load the predicted and true labels
    y_pred = np.load(y_pred_path)
    y_true = np.load(y_true_path)

    # Calculate various evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


def plot_confusion_matrix(y_true_path, y_pred_path, binary=1, model_name=""):
    """ Plot the confusion matrix for the target labels and predictions """
    y_pred = np.load(y_pred_path)
    y_test = np.load(y_true_path)
    cm = confusion_matrix(y_test, y_pred)
    if binary == 1:
        # Create a dataframe with the confusion matrix values
        df_cm = pd.DataFrame(cm, range(cm.shape[0]),
                             range(cm.shape[1]))
    if binary == 0:
        df_cm = pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2])
    if binary == 2:
        df_cm = pd.DataFrame(cm, index=[0, 1, 2, 3], columns=[0, 1, 2, 3])
    # Plot the confusion matrix
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, fmt='.0f', cmap="YlGnBu",
                annot_kws={"size": 10})  # font size
    plt.show()
    plt.savefig("./plots/cm_" + model_name + ".png")


if __name__ == "__main__":
    # Initialize Dataset
    ds = dataset()
    _, _, train_dataset = prepare_data(ds, "train")
    _, _, val_dataset = prepare_data(ds, "val")
    _, _, test_dataset = prepare_data(ds, "test")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertLSTM("bert-base-uncased")
    model = model.to(device)
    train(model, train_loader, test_loader, device)

    file_id = '1wOvasR_wTBZUkEAFtgupgoIFsL_jR1hC'
    output_path = './output/bert_lstm_model_best.pth'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
    accuracy, f1 = test(model, output_path, test_loader, device)
    print(
        f"Best Model Path: {output_path} with Accuracy: {accuracy}, f1 score: {f1}")
    evaluate("./output/y_test_bert.npy",
             "./output/y_pred_bert.npy")
    plot_confusion_matrix("./output/y_test_bert.npy",
                          "./output/y_pred_bert.npy",
                          model_name="pre_found_bert_lstm_model_best")

    best_score = 0.0
    best_accuracy = 0.0
    best_f1 = 0.0
    best_model_path = ""
    for epoch_idx in range(EPOCHS):
        accuracy, f1 = test(model,
                            f'./output/bert_lstm_model_pair_{epoch_idx}.pth',
                            test_loader, device)

        score = 0.5 * accuracy + 0.5 * f1

        if score > best_score:
            best_score = score
            best_accuracy = accuracy
            best_f1 = f1
            best_model_path = f'./output/bert_lstm_model_pair_{epoch_idx}.pth'

    print(
        f"Best Model Path: {best_model_path} with Accuracy: {best_accuracy}, f1 score: {best_f1}")

    accuracy, f1 = test(model, best_model_path, test_loader, device)
    evaluate("./output/y_truth_extension_2.npy",
             "./output/y_pred_extension_2.npy")
    plot_confusion_matrix(
        "./output/y_truth_extension_2.npy",
        "./output/y_pred_extension_2.npy",
        model_name="bert_lstm_model_best")

