import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#hyper paraameters

input_size= 784  #28*28
hidden_size= 100
num_classes=10
num_epochs= 5
batch_size=4
learning_rate=0.001

#mnist

train_dataset= torchvision.datasets.MNIST(root='./data',train=True,
                                          transform=transforms.ToTensor(),download=True)

test_dataset= torchvision.datasets.MNIST(root='./data',train=False,
                                          transform=transforms.ToTensor(),download=True)

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)


examples=iter(train_loader)
samples,labels=examples.next()
#print(samples,labels)

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(samples[i][0],cmap='gray')
plt.show()

class NerualNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NerualNet, self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return out

model = NerualNet(input_size,hidden_size,num_classes)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#training loop
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_loader):
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        # forward
        outputs=model(images)
        loss=criterion(outputs,labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0 :
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss={loss.item():.4f}')



with torch.no_grad():
  n_correct=0
  n_samples=0
  for images,labels in test_loader:
    images=images.reshape(-1,28*28).to(device)
    labels=labels.to(device)
    outputs=model(images)
    _,predictions=torch.max(outputs.data,1)
    n_samples+=labels.shape[0]
    n_correct+=(predictions ==labels).sum().item()
  acc = 100.0* n_correct /n_samples
  print(f'acc= {acc}')
