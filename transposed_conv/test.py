import torch
from torch import nn
from d2l import torch as d2l
#基础转置卷积
def  trans_conv(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]+h-1,X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i+h,j:j+h]+=X[i,j]*K
    return Y


X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(X)