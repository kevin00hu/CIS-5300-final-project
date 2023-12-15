
from dataset import dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import  Dataset, DataLoader
import gensim.downloader as api
import matplotlib.pyplot as plt
import nltk
import numpy as np
import string
import torch
import torch.nn as nn

torch.manual_seed(6666)
nltk.download("stopwords")

# loading dataset
FNC_PATH="./dataset/FNC-1"
LIAR_PATH="./dataset/LIAR/"
ds = dataset(FNC_PATH=FNC_PATH, LIAR_PATH=LIAR_PATH, word2vec=False)
train_df, val_df, test_df = ds(dataset="LIAR", all = True)


# loading word2vec embedding
word2vec_model = api.load("word2vec-google-news-300")


def get_embedding_map()->dict:
    embedding_map = {}

    stop = set(stopwords.words('english') + list(string.punctuation))
    temp = train_df['statement'].apply(lambda x: word_tokenize(x)).to_list()
    unique_words = sorted({j for i in temp for j in i if (j not in stop) and not j.isdigit()})

    for word in unique_words:
        if word in word2vec_model:
            embedding_map[word] = word2vec_model[word]

    return embedding_map

def get_embedding_matrix(embedding_map:dict, embedded_dim:int=300, randomize_init:bool = False):
    matrix_shape = (len(embedding_map)+1, embedded_dim)
    embedding_matrix = np.zeros( matrix_shape )

    for index, key in enumerate(sorted(embedding_map.keys())):
        embedding_matrix[index,:] = embedding_map.get(key, np.zeros(embedded_dim) if not randomize_init else np.random.normal(size=embedded_dim))
    
    # <UNK>
    embedding_matrix[index+1, :] = np.zeros(embedded_dim)

    return embedding_matrix

def create_embedding_layer(embedding_matrix, non_trainable=False):
    n_embed, d_embed = embedding_matrix.shape
    emb_layer = nn.Embedding(n_embed, d_embed, padding_idx = -1)
    emb_layer.weight = nn.Parameter( torch.tensor(embedding_matrix, requires_grad=non_trainable) )

    return emb_layer


embedding_map = get_embedding_map()
embedding_matrix = get_embedding_matrix(embedding_map, randomize_init=True)
word2index = {key: index for index, key in enumerate(sorted(embedding_map.keys()))}
index2word = {v:k for k,v in word2index.items()}
UNK_index = len(word2index)


def df_to_tensor_preprocess(df):
    X = df['statement'].apply(lambda x: np.array([word2index.get(i, UNK_index) for i in word_tokenize(x)]))

    length = torch.tensor([len(i) for i in X])
    largest_seq_length = max(length)
    
    X = torch.tensor([np.pad(i, (0,largest_seq_length-len(i)), constant_values = -1) for i in X])

    mask = torch.tensor(X != -1)
    X = X * mask

    y = df['label']
    y = torch.tensor(y)

    return X, y, mask, length

train_data = df_to_tensor_preprocess(train_df)
val_data   = df_to_tensor_preprocess(val_df)
test_data  = df_to_tensor_preprocess(test_df)


class LIAR_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, index):
        return [self.data[i][index] for i in range(4)]
    
train_dataset = LIAR_dataset(train_data)
val_dataset   = LIAR_dataset(val_data)
test_dataset  = LIAR_dataset(test_data)


batch_size = 1024

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)


device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTM_model(nn.Module):
    def __init__(self, 
                 d_embed=300,
                 d_hidden=150,
                 d_out=2,
                 embedding_matrix=None,
                 nl = 2,
                 bidirectional = True,
                 dropout = 0.4):
        super(LSTM_model, self).__init__()

        self.d_embed = d_embed
        self.d_hidden  = d_hidden
        self.d_out   = d_out
        self.embedding_matrix = embedding_matrix
        self.number_layer = nl
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embedding_layer = create_embedding_layer(embedding_matrix,False)
        self.LSTM = nn.LSTM(input_size = self.d_embed, 
                            hidden_size = self.d_hidden,
                            num_layers = self.number_layer,
                            batch_first = True,
                            bidirectional = self.bidirectional,
                            dropout = self.dropout)
        
        self.linear = nn.Linear(self.d_embed, 2)

    def forward(self, text, mask, length):
        max_length = max(length)

        text = text[:,:max_length]

        x = self.embedding_layer(text).to(torch.float32)
        extended_mask = mask.unsqueeze(-1).expand(-1, -1, 300)[:,:max_length,:]
        x = torch.multiply( x, extended_mask )

        packed_input = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM(packed_input)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=text.size(1))

        output = output.mean(axis=1)
        output = self.linear(output)

        return output


def train(lr:float = 0.001, epoch:int = 5, dropout=0.4, num_layers = 2):
    model = LSTM_model(embedding_matrix= embedding_matrix, dropout=dropout, nl = num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([sum(train_data[1]==1), sum(train_data[1]==0)]).to(device).float())

    best_loss  = float("inf")

    print("| epoch_number |  step | train loss | train accuracy | val loss | val accuracy | saved |")
    train_loss = []
    val_loss = []
    for epoch_number in range(1, epoch+1):
        step = 0
        model.train()
        save = False
        train_total_loss = 0
        for X, y, mask, length in train_loader:
            step += 1
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
            length = length.to(torch.int64)     
            optimizer.zero_grad()

            prediction = model(X, mask, length)
            loss = criterion(prediction, y)
            
            loss.backward()
            optimizer.step()

            train_total_loss += loss.detach().cpu() * X.size(0)
            accuracy = prediction.argmax(axis = 1) == y
            accuracy = sum(accuracy.detach().cpu().numpy())/len(accuracy.detach().cpu().numpy())
            print(f"\r| {epoch_number:12d} | {(step-1) * batch_size + X.size(0):5d} | {loss.item():10.4f} | {accuracy:14.4f} |", end="")
        train_loss.append(train_total_loss/len(train_data[0]))
        
        with torch.no_grad():
            model.eval()
            loss_list = []
            correct_list = []
            for X, y, mask, length in val_loader:
                X = X.to(device)
                y = y.to(device)
                mask = mask.to(device)
                length = length.to(torch.int64)
                val_prediction = model(X, mask, length)
                loss = criterion(val_prediction, y)
                loss_list.append(loss.item())

                val_accuracy = val_prediction.argmax(axis=1) == y
                val_accuracy = sum(val_accuracy.detach().cpu().numpy())
                correct_list.append(val_accuracy)

            current_loss = np.mean(loss_list)
            
            if  current_loss < best_loss:
                save = True
                best_loss =  current_loss
                torch.save(model, "LSTM_model")
            
            print(f" {np.mean(loss_list):8.4f} | {sum(correct_list)/len(val_data[0]):12.4f} | {'*' if save else ' ':^5s} |")
        val_loss.append(current_loss)
    return train_loss, val_loss


train_loss, val_loss = train(lr = 1e-3, epoch = 15, )


LSTM_model_PATH = "./LSTM_model"
model = torch.load(LSTM_model_PATH)
model.eval()

loss_list = []
correct_list = []
X, y, mask, length = test_data
X = X.to(device)
y = y.to(device)
mask = mask.to(device)
length = length.to(torch.int64)

test_prediction = model(X, mask, length)
test_prediction = test_prediction.argmax(axis=1).detach().cpu().numpy()

np.save("LSTM_prediction.npy", test_prediction)


plt.plot(range(len(train_loss)), train_loss, label = "train")
plt.plot(range(len(val_loss)), val_loss, label = "validation")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./plots/loss_LSTM.png')
plt.legend(['train',"validation"])
plt.show()


cm = confusion_matrix(test_data[1], test_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("./plots/cm_LSTM.png")

print(f"accuracy: {accuracy_score(test_data[1], test_prediction)} | f1_score: {f1_score(test_data[1], test_prediction)}")
