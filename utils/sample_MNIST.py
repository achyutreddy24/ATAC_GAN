import torch
from torchvision import datasets,transforms
from torchvision.utils import save_image

batch_size = 1
#output_dir = "MNIST_Samples/"

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=False,
)

i = 0
for data in dataloader:
    img, labels = data
    save_image(torch.FloatTensor(img.data), "mnist_images/" + str(i) + ".png", nrow=1, normalize=True)
    i+=1
    if (i > 100):
        break
