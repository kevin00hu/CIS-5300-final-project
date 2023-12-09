import os
import numpy as np
import argparse
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


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
    plt.savefig("./plots/cm_"+model_name+".png")



def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Evaluate classification predictions.')
    parser.add_argument('y_true_path', type=str,
                        help='Path to the .npy file with true labels')
    parser.add_argument('y_pred_path', type=str,
                        help='Path to the .npy file with predicted labels')

    # Parse the command-line arguments
    args = parser.parse_args()

    model_name = os.path.splitext(os.path.basename(args.y_pred_path))[0].split('_')[-1]


    # Run the evaluation
    evaluate(args.y_true_path, args.y_pred_path)
    plot_confusion_matrix(args.y_true_path, args.y_pred_path, model_name=model_name)


if __name__ == '__main__':
    main()
