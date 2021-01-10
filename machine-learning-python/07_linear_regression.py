import torch
import numpy as np
from sklearn import datasets
import torch.nn as nn
import matplotlib.pyplot as plt


#prepare data
x_numpy, y_numpy= datasets.make_regression(n_samples=100,n_features=1,noise=20)

x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(-1,1)
print(x.shape,y.shape)

num_sample,n_features=x.shape

#model

input_size=n_features
output_size=1

model=nn.Linear(input_size,output_size)
#loss
criterion=nn.MSELoss()

#optimizer
lr=0.001
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

#training_loop
epochs=1000
for epoch in range(epochs):
    y_pred=model(x)

    loss=criterion(y_pred,y)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)% 10 ==0:
        print(f'epoch {epoch+1} , loss= {loss.item():.4f}')


predicted=model(x).detach().numpy()
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
plt.show()