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
from MNIST_Generators import Generator_3Ca as Generator
from MNIST_Classifiers import Classifier_4Ca as Classifier

# Argument Parsing
#TBD


# Config

cuda = True
training_set_size = 60000
n_epochs=4000
batch_size=64
gen_batch_size=64
lr=0.0002
b1=0.5
b2=0.999
n_classes=10
latent_dim=100
img_size=28
channels=1
save_interval=1000
print_interval=1000
test_interval=2000

real_loss_coeff = 0.7
gen_loss_coeff = 0.3

load_c = True
load_c_path = "../output/MNIST-C60000-2290918716792143116/C"
load_g_path = "../output/MNIST-9721266708110142777/G"

output_dir="../output/MNIST-CuG-" + str(torch.random.initial_seed())


# Util

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def load_model(m, o, path):
    d = torch.device("cuda" if cuda else "cpu")
    load = torch.load(path, map_location=d)
    m.load_state_dict(load["model_state_dict"])
    if (o):
        o.load_state_dict(load["optimizer_state_dict"])
    m.to(d)
    

if __name__ == "__main__":
    # Loss function
    loss = nn.CrossEntropyLoss()


    # Initialize/Load Models
    c = Classifier()
    g = Generator(latent_dim)

    optimizer = torch.optim.Adam(c.parameters(), lr=lr, betas=(b1, b2))

    if cuda:
        c.cuda()
        loss.cuda()
        g.cuda()
        
    if (load_c):
        load_model(c, optimizer, load_c_path)
    else:
        c.apply(init_weights)
    
    load_model(g, None, load_g_path)
    g.eval()
    for param in g.parameters():
        param.requires_grad = False


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
    f.write("real_loss_coeff: {}\n".format(real_loss_coeff))
    f.write("gen_loss_coeff: {}\n".format(gen_loss_coeff))
    f.write("load_g_path: {}\n".format(load_g_path))
    f.write("load_c_path: {}\n".format(load_c_path))
    f.close()

    f = open(output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    log_writer.writerow(['Epoch', 'Batch', 'Loss', 'LossReal', 'LossGen', 'AccReal', 'AccGen'])

    running_acc_real = 0.0
    running_acc_gen = 0.0
    running_loss = 0.0
    running_loss_real = 0.0
    running_loss_gen = 0.0

    
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
            
            # =============
            #  Real Images
            # =============
            
            batch_size = imgs.shape[0]
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            
            optimizer.zero_grad()

            pred = c(real_imgs)
            l_real = loss(pred, labels)
            
            
            # ==================
            #  Generated Images
            # ==================

            z = Variable(FloatTensor(np.random.normal(0, 1, (gen_batch_size, latent_dim))))
            g_labels = Variable(LongTensor(np.random.randint(0, 10, gen_batch_size)))
            g_target_labels = Variable(LongTensor(np.random.randint(0, 10, gen_batch_size)))

            # Generate images
            gen_imgs = g(z, g_labels, g_target_labels)
            
            gen_pred = c(gen_imgs)
            l_gen = loss(gen_pred, g_labels)
            
            l = real_loss_coeff * l_real + gen_loss_coeff * l_gen 
            l.backward()
            optimizer.step()
                

            # =========
            #  Logging
            # =========

            running_loss += l.item()
            running_loss_real += l_real.item()
            running_loss_gen += l_gen.item()
            acc_real = np.mean(np.argmax(pred.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy()) * 100
            acc_gen = np.mean(np.argmax(gen_pred.detach().cpu().numpy(), axis=1) == g_labels.detach().cpu().numpy()) * 100
            running_acc_real += acc_real
            running_acc_gen += acc_gen

            log_writer.writerow([epoch, i, l.item(), l_real.item(), l_gen.item(), acc_real, acc_gen])


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
                print("Epoch %d/%d, Batch %d/%d -- LOSS: %f (Real: %f, Gen: %f), Acc Real: %.2f%%, Acc Gen: %.2f%%" % (epoch, n_epochs, i, len(dataloader), running_loss / p, running_loss_real / p, running_loss_gen / p, running_acc_real / p, running_acc_gen / p))

                running_acc_real = 0.0
                running_acc_gen = 0.0
                running_loss = 0.0
                running_loss_real = 0.0
                running_loss_gen = 0.0

            # Save sample images
            if batches_done % test_interval == 0:
                test()
