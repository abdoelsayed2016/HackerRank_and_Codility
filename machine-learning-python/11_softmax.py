import torch
import torch.nn as nn
import  numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x=np.array([2,3,1])
output = softmax(x)
print(output)


x=torch.tensor([2.0,3.0,1.0])
output=torch.softmax(x,dim=0)
print(output)


class NerualNetwork(nn.Module):
    def __init__(self,input_dim,hidden_size,output_dim):
        self.linear1=nn.Linear(input_dim,hidden_size)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_size,output_dim)
    def forward(self,x):
        out=self.linear1(x)
        out=self.relu(out)
        out=self.linear2(out)
        return out


model = NerualNetwork(input_dim=28*28,hidden_size=5,output_dim=3)

criterion=nn.CrossEntropyLoss()

