import argparse
import os
import numpy as np
import math
import csv

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# Models
from torchvision.models import vgg19
from sys import path,maxsize
path.append("../utils")
from models import LeNet5


# Argument Parsing
#TBD


# Config

cuda = False
training_set_size = 5000
n_epochs=200
batch_size=24
lr=0.0002
b1=0.5
b2=0.999
n_classes=10
img_size=28
channels=1
save_interval=200
print_interval=200
test_interval=500

output_dir="../output/MNIST-AC-" + str(torch.random.initial_seed())


# Util

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Models

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def classifier_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *classifier_block(channels, 16, bn=False),
            *classifier_block(16, 32),
            *classifier_block(32, 64),
            *classifier_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 2#img_size // 2 ** 4

        # Output layer
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        label = self.aux_layer(out)

        return label

if __name__ == "__main__":
    # Loss function
    loss = nn.CrossEntropyLoss()


    # Initialize/Load Models
    c = Classifier()
    c.apply(init_weights)

    optimizer = torch.optim.Adam(c.parameters(), lr=lr, betas=(b1, b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if cuda:
        c.cuda()
        loss.cuda()


    # Configure data loader
    
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    os.makedirs("../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data/mnist",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(list(range(training_set_size)))
    )
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


    # Set up logging/saving

    os.mkdir(output_dir)

    f = open(output_dir + '/constants.txt', 'a')
    f.write("cuda: {}\n".format(cuda))
    f.write("batch_size: {}\n".format(batch_size))
    f.write("lr: {}\n".format(lr))
    f.write("adam_b1: {}\n".format(b1))
    f.write("adam_b2: {}\n".format(b2))
    f.write("n_classes: {}\n".format(n_classes))
    f.write("img_size: {}\n".format(img_size))
    f.write("channels: {}\n".format(channels))
    f.write("save_interval: {}\n".format(save_interval))
    f.write("print_interval: {}\n".format(print_interval))
    f.write("test_interval: {}\n".format(test_interval))
    f.write("output_dir: {}\n".format(output_dir))
    f.close()

    f = open(output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    log_writer.writerow(['Epoch', 'Batch', 'Loss', 'Acc'])

    running_acc = 0.0
    running_loss = 0.0

    
    def test():
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                out = c(images)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.2f%%' % (100 * correct / total))

    # Train

    for epoch in range(n_epochs):

        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            
            optimizer.zero_grad()

            pred = c(real_imgs)
            l = loss(pred, labels)

            l.backward()
            optimizer.step()


            # =========
            #  Logging
            # =========

            running_loss = l.item()
            acc = np.mean(np.argmax(pred.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy()) * 100
            running_acc += acc

            log_writer.writerow([epoch, i, l.item(), acc])


            batches_done = epoch * len(dataloader) + i

            # Save models
            if batches_done % save_interval == 0:
                # Saves weights
                torch.save(c.state_dict(), output_dir + '/C')

            # Print information
            if batches_done % print_interval == 0:

                p = float(print_interval)
                print("Epoch %d/%d, Batch %d/%d -- LOSS: %f, ACC: %.2f%%" % (epoch, n_epochs, i, len(dataloader), running_loss / p, running_acc / p))

                running_acc = 0.0
                running_loss = 0.0

            # Save sample images
            if batches_done % test_interval == 0:
                test()
