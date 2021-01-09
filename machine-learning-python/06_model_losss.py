import torch

import torch.nn as nn


x= torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

n_samples,n_features=x.shape

print(n_samples,n_features)

input_size=n_features

output_size=n_features

model=nn.Linear(input_size,output_size)

print(f'prediction before training: f(5) = {model(torch.tensor([5],dtype=torch.float32)).item():0.3f}')

lr=0.01
n_iters=1000

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

for epoch in range(n_iters):
    y_pred=model(x)
    l=loss(y,y_pred)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch %10 == 0:
        [w,b]= model.parameters()
        print(w.item(),b.item())
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss={l:0.8f}')




print(f'prediction after training: f(5) = {model(torch.tensor([5],dtype=torch.float32)).item():0.3f}')

