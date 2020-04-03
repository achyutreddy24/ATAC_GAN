import torch
from torchvision import datasets,transforms

n = 5000
batch_size = 64
#output_dir = "MNIST_Samples/"

transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=False,
        download=True,
        transform=transform,
    ),
    batch_size=batch_size,
    sampler=torch.utils.data.SubsetRandomSampler(list(range(n)))
)

totalLabels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for data in test_dataloader:
    _, labels = data
    for label in labels:
        #print(label.item())
        totalLabels[label.item()] += 1

print(totalLabels)