from dataset import dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize):
        super(linearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, 256)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 64)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(64,32)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(32,2)
        # self.activation4 = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.activation1(self.linear1(x))
        out = self.activation2(self.linear2(out))
        out = self.activation3(self.linear3(out))
        out = self.linear4(out)
        # out = self.activation4(self.linear4(out))
        return out


ds = dataset()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # only use LIAR statment vectors (None, 300)
    train_df = ds(dataset = "LIAR", type = "train")
    train_X = train_df[[i for i in train_df.columns if "vector_" in i]]
    train_y = train_df["label"]

    val_df = ds(dataset = "LIAR", type = "val")
    val_X = val_df[[i for i in val_df.columns if "vector_" in i]]
    val_y = val_df["label"]

    test_df = ds(dataset = "LIAR", type = "test")
    test_X = test_df[[i for i in test_df.columns if "vector_" in i]]
    test_y = test_df["label"]

    

    train = torch.utils.data.TensorDataset(torch.tensor(train_X.values), torch.tensor(train_y.values))
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)

    val_X, val_y = torch.tensor(val_X.values), torch.tensor(val_y.values)
    val_X, val_y = val_X.to(device), val_y.to(device)

    test_X, test_y = torch.tensor(test_X.values), torch.tensor(test_y.values)
    test_X, test_y = test_X.to(device), test_y.to(device)


    model = linearRegression(train_X.shape[1]).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_records = []
    val_loss_records = []
    epoches = 10
    for epoch in range(epoches):
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred.float(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_records.append(loss.cpu().detach())

        with torch.no_grad():
            pred = model(val_X)
            val_loss = criterion(pred.float(), val_y)
            val_accuracy = accuracy_score(val_y.cpu(), np.argmax(pred.cpu().numpy(),axis=1))
            print(f"""Epoch: {epoch}, train loss: {loss:.4f}, val loss: {val_loss:.4f}
                val accuracy: {val_accuracy:.4f}""")

            val_loss_records.append(val_loss.cpu().detach())

    with torch.no_grad():
        test_pred = model(test_X)
        test_y, test_pred = test_y.cpu(), np.argmax(test_pred.cpu().numpy(),axis=1)
        test_accuracy = accuracy_score(test_y, test_pred)
        test_f1 = f1_score(test_y, test_pred)
        print(f"test accuracy: {test_accuracy} | test f1 score: {test_f1}")

    np.save("./output/y_test_baseline.npy", test_y)
    np.save("./output/y_pred_baseline.npy", test_pred)

    cm = confusion_matrix(test_y, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("./plots/baseline_cm.png")
    plt.clf()


    plt.plot(range(len(train_loss_records)), train_loss_records, label = "train")
    plt.plot(range(len(val_loss_records)), val_loss_records, label = "validation")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train',"validation"])
    plt.savefig('./plots/baseline_loss.png')
    plt.show()
    
#+----------------------------------------------+
# command:
# python3 baseline.py
# or 
# python3 -m baseline
#+----------------------------------------------+