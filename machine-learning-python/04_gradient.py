import torch


x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([3,4,5,6],dtype=torch.float32)

w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)


def forward(x):
    return x*w

def loss(y,y_pred):
    return  ((y_pred-y)**2).mean()

lr=0.03
epochs=30

for epoch in range(epochs):
    y_pred=forward(x)

    l=loss(y,y_pred)

    l.backward()

    with torch.no_grad():
        w-=lr*w.grad

    w.grad.zero_()

    if epoch % 2 == 0 :
        print(f'epoch {epoch+1}: w = {w:.3f}, loss ={l:.8f} ')
