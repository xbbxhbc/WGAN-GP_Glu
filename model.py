from ReliableWGANGP import ReliableWGAN
from Dataprocess import Testprocess
import torch.utils.data as Data
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
np.random.seed(23)
torch.cuda.manual_seed(42)
# Attention
class BahdanauAttention(nn.Module):

    def __init__(self, in_features, hidden_units, num_task):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)

        score = self.V(nn.Tanh()(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values, 1, 2)  # transpose to make it suitable for matrix multiplication

        context_vector = torch.matmul(values, attention_weights)
        context_vector = torch.transpose(context_vector, 1, 2)
        return context_vector, attention_weights


class MyDeep(nn.Module):
    def __init__(self):
        super(MyDeep, self).__init__()
        kernel_size = 10
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  # input height
                out_channels=256,  # n_filters
                kernel_size=kernel_size),  # filter size
            nn.ReLU(),  # activation
            nn.BatchNorm1d(256),
            nn.Dropout())
        self.lstm = torch.nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True)  #
        self.Attention = BahdanauAttention(in_features=256, hidden_units=10, num_task=1)
        self.fc_task = nn.Sequential(
            nn.Linear(256, 698),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(698, 1250),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1250, 2),
        )
        self.classifier = nn.Linear(2, 1)

    # ---------------->>> Forward
    def forward(self, x):
        batch_size, features, seq_len = x.size()

        x = self.conv1(x)
        x = x.transpose(1, 2)
        # rnn layer
        out, (h_n, c_n) = self.lstm(x)

        h_n = h_n.view(batch_size, out.size()[-1])
        context_vector, attention_weights = self.Attention(h_n, out)

        reduction_feature = self.fc_task(torch.mean(context_vector, 1))

        representation = reduction_feature
        logits_clsf = self.classifier(representation)
        logits_clsf = torch.sigmoid(logits_clsf)
        return logits_clsf
#metric
def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    mcc = matthews_corrcoef(y.cpu().detach().numpy(), rounded_preds.cpu().detach().numpy())
    return acc,mcc
def metric(preds, y):
    a = preds.cpu().numpy()
    rounded_preds = torch.round(preds)
    TN, FP, FN, TP = confusion_matrix(y.cpu().numpy(), rounded_preds.cpu().numpy()).ravel()
    SN = recall_score(y.cpu().numpy(), rounded_preds.cpu().numpy())
    SP = TN / (TN + FP)
    MCC = matthews_corrcoef(y.cpu().numpy(), rounded_preds.cpu().numpy())
    ACC = (TP + TN) / (TP + TN + FN + FP)
    return SN, SP, ACC, MCC,a
#training
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    te_loss=0
    model.train()
    for i, batch in enumerate(iterator, 0):
        x_data, x_label = batch
        optimizer.zero_grad()
        x_data = x_data.unsqueeze(1).float()
        predictions = model(x_data.cuda()).squeeze(1)
        loss = criterion(predictions, x_label.cuda().float())
        acc,_ = binary_accuracy(predictions, x_label.cuda().float())
        loss.backward()
        learn_te = optimizer.state_dict()['param_groups'][0]['lr']
        optimizer.step()
        te_loss+=loss
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator),learn_te
#tesing
def Test(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_SN = 0
    epoch_SP = 0
    epoch_ACC = 0
    epoch_MCC = 0
    model.eval()
    with torch.no_grad():
        cnn_pro=[]
        cnn_y=[]
        new_test_loss=[]
        new_test_acc=[]
        for i, batch in enumerate(iterator, 0):
            test, t_lable = batch
            cnn_y=cnn_y+t_lable.tolist()
            test = test.unsqueeze(1).float()
            predictions = model(test.cuda()).squeeze(1)
            loss = criterion(predictions, t_lable.cuda().float())
            acc,_ = binary_accuracy(predictions, t_lable.cuda())
            SN, SP, ACC, MCC,CNN_DNN_Pro = metric(predictions, t_lable.cuda())
            CNN_DNN_Pro=CNN_DNN_Pro.tolist()
            cnn_pro=cnn_pro+CNN_DNN_Pro
            new_test_acc.append(acc.item())
            new_test_loss.append(loss.item())
            epoch_SN += SN
            epoch_SP += SP
            epoch_ACC += ACC
            epoch_MCC += MCC
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_SN / len(iterator), epoch_SP / len(
        iterator), epoch_ACC / len(iterator), epoch_MCC / len(iterator),cnn_pro,cnn_y,new_test_loss,new_test_acc
import time
#static time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
#loading train set
X_train,y_train=ReliableWGAN()
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
#loading test set
X_test,y_test=Testprocess()
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
torch.manual_seed(40)
BATCH_SIZE =64
train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)
train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
torch.manual_seed(20)
test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
import torch.optim as optim
N_EPOCHS = 600
#Initialize the object
model = MyDeep().cuda()
optimizer=optim.SGD(model.parameters(),lr=0.09811)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.cuda()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')
print('---------DNN_CNN_Training-------')
new_train_loss=[]
new_train_acc=[]
new_learn=[]
best_epoch=0
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc,learn_train_te = train(model, train_loader, optimizer, criterion)
    new_learn.append(learn_train_te)
    new_train_loss.append(train_loss)
    new_train_acc.append(train_acc)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
model.load_state_dict(torch.load('./parameter/model.pt'))
print('---------DNN_CNN_Testing-------')
test_loss, test_acc, test_sn, test_sp, test_ACC, test_mcc,CNN_DNN_Pro,cnn_y,new_test_loss,new_test_acc= Test(model, test_loader, criterion)
print(f'test_sn: {test_sn*100:.2f}% | test_sp: {test_sp*100:.2f}%')
print(f'test_mcc: {test_mcc:.4f} | test_ACC: {test_ACC * 100:.2f}%')




