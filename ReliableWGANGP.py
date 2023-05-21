from torch import nn, optim, autograd
import torch
import torch.optim as optim
import numpy as np
h_dim = 220
batchsz = 590
torch.manual_seed(23)
np.random.seed(23)
# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # z:
            nn.Linear(76, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 76),
        )

    def forward(self, z):
        output = self.net(z)
        return output

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(76, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 200),  # 190
            nn.ReLU(True),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

#gradient penalty
def gradient_penalty(D, xr, xf):
    t = torch.rand(batchsz, 1).cuda()
    t = t.expand_as(xr)
    mid = t * xr + (1 - t) * xf
    mid.requires_grad_()
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid, grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp
def ReliableWGAN():
    import pandas as pd
    import torch
    torch.manual_seed(23)
    np.random.seed(23)
    #Read the training set
    fpath = "./data/train.txt"
    train = pd.read_csv(
        fpath,
        sep=' ',
        header=None,
    )
    #read the positive samples
    da = []
    for i in range(590):
        aa = train.loc[i, 0:75].tolist()
        aa = list(map(float, aa))
        da.append(aa)
    # read the unlable samples
    import pandas as pd
    fpath = "./data/train.txt"
    train = pd.read_csv(
        fpath,
        sep=' ',
        header=None,
    )
    ii = 590
    unlabel = []
    while ii < 4088:
        aa = train.loc[ii, 0:75].tolist()
        aa = list(map(float, aa))
        unlabel.append(aa)
        ii += 1
    #positive samples
    X_train_pos = da[0:590]
    da = X_train_pos
    #Stores randomly selected unlabeled samples
    X_random1 = []
    import random
    random.seed(22)
    for i in range(590):
        X_random1.append(unlabel[random.randint(0, 3497)])
    # Stores randomly selected unlabeled samples
    random.seed(23)
    X_random2 = []
    for i in range(590):
        X_random2.append(unlabel[random.randint(0, 3497)])
    # ---------------------------------selecting reliable negative samples---------------------------
    data_pos = np.array(da).astype(np.float32)
    # list->numpy
    data_neg = np.array(unlabel).astype(np.float32)
    X_random1 = np.array(X_random1).astype(np.float32)
    X_random2 = np.array(X_random2).astype(np.float32)
    # numpy->tensor
    data_neg = torch.from_numpy(data_neg)
    data_pos = torch.from_numpy(data_pos)
    X_random1 = torch.from_numpy(X_random1)
    X_random2 = torch.from_numpy(X_random2)
    #Initialize the generator
    G1 = Generator().cuda()
    G2 = Generator().cuda()
    # Initialize the Discriminator
    D = Discriminator().cuda()
    G1.apply(weights_init)
    G2.apply(weights_init)
    D.apply(weights_init)
    optim_G1 = optim.SGD(G1.parameters(), lr=0.001)
    optim_G2 = optim.SGD(G2.parameters(), lr=0.006)
    optim_D = optim.SGD(D.parameters(), lr=0.01)  # 0.0015
    learning_d = []
    learning_g = []
    for epoch in range(1000):
        # Ⅰ。training Discriminator
        for _ in range(5):
            #Positive samples as real data
            xr = data_pos.cuda()
            predr = D(xr)
            lossr = -(predr.mean())
            #Positive samples are input into the generator to generate data
            z = data_pos.cuda()
            xf = G1(z).detach()
            predf = D(xf)
            lossf1 = -(predf.mean())
            # unlabeled samples are input into the generator to generate data
            z1 = X_random1.cuda()
            xf1 = G2(z1).detach()
            predf2 = D(xf1)
            lossf2 = predf2.mean()
            #gradient penalty
            gp = gradient_penalty(D, xr, xf.detach())
            # loss_D
            loss_D = lossr + lossf1 - lossf2 + (10 * gp)
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
        learn = optim_D.state_dict()['param_groups'][0]['lr']
        learning_d.append(learn)
        #2.training Generator
        z1 = X_random1.cuda()
        xf1 = G2(z1)
        predf1 = D(xf1)
        z2 = X_random2.cuda()
        xf2 = G1(z2)
        predf2 = D(xf2)
        loss_G1 = -(predf1.mean())
        loss_G2 = -(predf2.mean())
        optim_G1.zero_grad()
        optim_G2.zero_grad()
        loss_G1.backward()
        loss_G2.backward
        learn1 = optim_G1.state_dict()['param_groups'][0]['lr']
        learning_g.append(learn1)
        optim_G1.step()
        optim_G2.step()
    #-----selecting negative samples---
    with torch.no_grad():
        a = data_neg
        i = 0
        num_pos = 0
        num_neg = 0
        neg_position = []
        while i < len(a):
            pred_un = D(a[i].cuda())
            if pred_un >= 0.5:
                num_pos = num_pos + 1
            if pred_un < 0.5:
                neg_position.append(i)
                num_neg = num_neg + 1
            i = i + 1

    # Store reliable negative samples
    reliable_neg = []
    i = 0
    while (i < len(neg_position)):
        reliable_neg.append(unlabel[neg_position[i]])
        i = i + 1
    # print("neg",len(reliable_neg))
    # get train set
    X_train = X_train_pos + reliable_neg
    # print('len(X_train)', len(X_train))
    #geting lable
    y_train = []
    i = 0
    while (i < len(X_train)):
        if (i < len(X_train_pos)):
            y_train.append(1)
        else:
            y_train.append(0)
        i = i + 1
    # list->array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train,y_train



