import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


dc = datasets.load_breast_cancer()
x,y=dc.data,dc.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.2)

tc=StandardScaler()
x_train=tc.fit_transform(x_train)
x_test=tc.fit_transform(x_test)

x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(-1,1)
y_test=y_test.view(-1,1)


learning_rate=0.01
epochs=1000

n_samples,n_features=x.shape
class LogisticRegression(nn.Module):
    def __init__(self,n_features):
        super(LogisticRegression, self).__init__()
        self.linear=nn.Linear(n_features,1)
    def forward(self,x):
        return torch.sigmoid(self.linear(x))

model=LogisticRegression(n_features)

criterion=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    y_pred=model(x_train)
    loss=criterion(y_pred,y_train)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f'epoch {epoch+1}, loss: {loss.item():.4f}')


with torch.no_grad():
    y_pred=model(x_test)
    y_pred_cls=y_pred.round()
    acc = y_pred_cls.eq(y_test).sum()/ float(y_test.shape[0])
    print(f'acc {acc:.4f}')