import numpy as np
import matplotlib.pyplot as plt
epochs=100
lr=0.001
m=10
feature=1
x=np.random.rand(m,feature)
y=np.random.rand(m,1)

w=np.random.rand(feature,1)
b=np.random.random()

cost_log=[]
for i in range(epochs):
    pred=np.matmul(x,w)+b
    cost= 1/(2*m) * np.sum(np.square(np.subtract(pred,y)))
    if i % (epochs//100) ==0:
        print(i // (epochs // 100), "%")
        print(cost)
        cost_log.append(cost)
    w_gradient=1/m * np.sum(np.multiply(np.subtract(pred,y),x))
    b_gradient=1/m * np.sum(np.subtract(pred,y))

    w-=lr* w_gradient
    b-= lr * b_gradient

plt.figure()
plt.title('graph of train data')

plt.scatter(x,y)
pred=np.matmul(x,w)+b
plt.plot(x,pred)
plt.xticks(())
plt.yticks(())
plt.show()


print(b)