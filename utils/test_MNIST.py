from sys import path
import os
path.append("../training")
import ATACGAN_MNIST_Split as atacgan
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms,datasets

parser = argparse.ArgumentParser("Test MNIST Classifier Performance")
parser.add_argument("--model_dir", type=str, help="Directory with model C present")
parser.add_argument("--cuda", default=True, action="store_false", help="Use cuda")
args = parser.parse_args()

batch_size = 24

if (args.model_dir[len(args.model_dir)-1] != '/'):
    args.model_dir += '/'

if (args.cuda):
    print("Cuda is set to True, use --cuda to set False")
    
classifier = atacgan.Classifier()
classifier.load_state_dict(torch.load(args.model_dir + "C", map_location=torch.device('cuda' if args.cuda else 'cpu')))
classifier.eval()

transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=False,
        download=True,
        transform=transform,
    ),
    batch_size=batch_size,
    shuffle=True
)

if args.cuda:
    classifier.cuda()

total = 0
correct = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        out = classifier(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f%%' % (100.0 * float(correct) / float(total)))