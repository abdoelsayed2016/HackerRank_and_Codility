import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)

epochs= 2
total_sample= len(dataset)
n_iterations=math.ceil(total_sample/4)
print(n_iterations,total_sample)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):

        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{epochs} , step {i + 1}/{n_iterations}, inputs {inputs.shape}  ')
