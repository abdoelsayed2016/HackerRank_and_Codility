import numpy as np


x=np.array([1,2,3,4],dtype=np.float32)
y=np.array([3,5,6,7],dtype=np.float32)

w=0.0


def foward(x):
    return x * w
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()
def gradient(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()

epochs=10
lr=.01

#print()

print(f'before training f(5) : {foward(5):.3f}')
for epoch in range(epochs):
    y_pred=foward(x)

    l=loss(y,y_pred)

    dw=gradient(x,y,y_pred)

    w-=dw*lr

    print(f'epoch {epoch+1} w= {w:.3f}, loss ={l:.8f}')

print(f'before training f(5) : {foward(5):.3f}')
