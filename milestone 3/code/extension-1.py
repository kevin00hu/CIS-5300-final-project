# encoding=utf-8
from math import inf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    get_linear_schedule_with_warmup

import gdown
from dataset import dataset

torch.cuda.manual_seed(0)

EPOCHS = 5


# Prepare Data
def prepare_data(ds, dataset_type):
    data_list = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = ds(dataset="LIAR", type=dataset_type)

    statements = df["statement"].values.tolist()
    contexts = df["context(location)"].values.tolist()

    inputs = [f"{statement}{tokenizer.sep_token}{context}{tokenizer.sep_token}"
              for statement, context in zip(statements, contexts)]
    X = tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                    padding="max_length", truncation=True,
                                    return_tensors="pt")

    y = df["label"].values.tolist()

    negative_num = 0
    for idx, label in enumerate(y):
        if label == 0:
            negative_num += 1

        data = {
            "input_ids": X["input_ids"][idx],
            "attention_mask": X["attention_mask"][idx],
            "token_type_ids": X["token_type_ids"][idx],
            "labels": torch.tensor(label, dtype=torch.int64)
        }
        data_list.append(data)

    return X, y, data_list


def train(model,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          device,
          epochs: int = EPOCHS):
    best_f1_score = 0.0
    train_loss_records = []
    val_loss_records = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8,
                                  weight_decay=0.01)
    scheduled = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_dataloader) * epochs * 0.1,
        num_training_steps=len(train_dataloader) * epochs)

    val_loss_improve_nums = 0
    early_stop_patience = 3
    model.train()

    last_val_loss = inf
    for epoch_idx in range(epochs):
        total_train_loss = 0.0
        total_val_loss = 0.0
        for train_batch in tqdm(train_dataloader):

            for key in train_batch:
                train_batch[key] = train_batch[key].to(device)

            loss = model(**train_batch).loss
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

        with torch.no_grad():
            total_truth_labels = []
            total_pred_labels = []
            for idx, val_batch in enumerate(val_dataloader):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)
                val_results = model(**val_batch)
                logits = val_results.logits
                val_loss = val_results.loss.item()
                total_val_loss += val_loss
                logits_after_softmax = logits.softmax(dim=1).cpu().detach()

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

        print(f"epoch-num: {epoch_idx}, "
              f"train-loss: {train_loss_records[-1]}, "
              f"val-loss: {val_loss_records[-1]}")
        print(f"epoch-num: {epoch_idx}, "
              f"average-accuracy: {average_accuracy_score}, "
              f"average-precision: {average_precision_score}, "
              f"average-recall: {average_recall_score}, "
              f"average-f1: {average_f1_score}")

        torch.save(model.state_dict(),
                   f'./output/bert_model_pair_new_{epoch_idx}.pth')

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
    plt.savefig('./plots/bert_loss.png')
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

            test_results = model(**test_batch)
            logits = test_results.logits
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

    np.save("./output/y_test_bert.npy",
            total_truth_labels)
    np.save("./output/y_pred_bert.npy", pred_truth_labels)

    print(f"accuracy: {accuracy}, "
          f"precision: {precision}, "
          f"recall: {recall}, f1: {f1}")

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
    # device = "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, num_hidden_layers=2)
    model = model.to(device)
    train(model, train_loader, val_loader, device)

    file_id = '1-SfBRZo9bISNgnm2c9lgqJOrGkjurqFM'
    output_path = '/content/drive/MyDrive/output/bert_model_best.pth'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
    accuracy, f1 = test(model, output_path, test_loader, device)
    print(
        f"Best Model Path: {output_path} with Accuracy: {accuracy}, f1 score: {f1}")
    evaluate("./output/y_test_bert.npy",
             "./output/y_pred_bert.npy")
    plot_confusion_matrix("./output/y_test_bert.npy",
                          "./output/y_pred_bert.npy",
                          model_name="pre_found_bert_model_best")

    best_score = 0.0
    best_accuracy = 0.0
    best_f1 = 0.0
    best_model_path = ""
    for epoch_idx in range(EPOCHS):
        accuracy, f1 = test(model,
                            f'./output/bert_model_pair_new_{epoch_idx}.pth',
                            test_loader, device)

        score = 0.5 * accuracy + 0.5 * f1

        if score > best_score:
            best_score = score
            best_accuracy = accuracy
            best_f1 = f1
            best_model_path = f'./output/bert_model_pair_new_{epoch_idx}.pth'

    print(f"Best Model Path: {best_model_path} with Accuracy: {best_accuracy}, f1 score: {best_f1}")

    accuracy, f1 = test(model, best_model_path, test_loader, device)
    evaluate("./output/y_test_bert.npy",
             "./output/y_pred_bert.npy")
    plot_confusion_matrix("./output/y_test_bert.npy",
                          "./output/y_pred_bert.npy",
                          model_name="bert_model_best")
