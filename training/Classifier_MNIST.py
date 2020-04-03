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
from sys import path
path.append("../models")
from MNIST_Classifiers import Classifier_4Ca as Classifier


# Argument Parsing
#TBD


# Config

cuda = True
training_set_size = 60000
n_epochs=600
batch_size=64
lr=0.0002
b1=0.5
b2=0.999
n_classes=10
img_size=28
channels=1
save_interval=1000
print_interval=1000
test_interval=1000

output_dir="../output/MNIST-C" + str(training_set_size) + "-" + str(torch.random.initial_seed())


# Util

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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
    f.write("training_set_size: {}\n".format(training_set_size))
    f.write("output_dir: {}\n".format(output_dir))
    f.close()

    f = open(output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    log_writer.writerow(['Epoch', 'Batch', 'Loss', 'Acc'])

    running_acc = 0.0
    running_loss = 0.0

    
    def test():
        c.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images = Variable(images.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))
                out = c(images)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy on the test set: %.2f%%' % (100 * correct / total))
        c.train()

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

            # Save model
            if batches_done % save_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": c.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": l
                }, output_dir + '/C')

            # Print information
            if batches_done % print_interval == 0:

                p = float(print_interval)
                print("Epoch %d/%d, Batch %d/%d -- LOSS: %f, ACC: %.2f%%" % (epoch, n_epochs, i, len(dataloader), running_loss / p, running_acc / p))

                running_acc = 0.0
                running_loss = 0.0

            # Save sample images
            if batches_done % test_interval == 0:
                test()
