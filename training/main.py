import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
        shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(labels)

net = models.wide_resnet50_2().cuda()

# Defining loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    print("EPOCH", epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/2000))
        running_loss = 0.0

print("DONE!")

PATH = "../models/test/cifar10-resnet"
torch.save(net.state_dict(), PATH)
