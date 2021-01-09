import torch
import torch.nn as nn

x= torch.tensor([1,2,3,4])
y=torch.tensor([2,4,6,8])

w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

def forward(x):
    return w*x


lr=0.01
n_iters=10000


loss=nn.MSELoss()
optimizer=torch.optim.SGD([w],lr=lr)


for epoch in range(n_iters):
    y_pred=forward(x)

    l=loss(y,y_pred)

    l.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch : {epoch+1}: w={w:.3f}, loss={l:.8f}')