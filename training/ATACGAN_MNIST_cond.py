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
from sys import path
path.append("../utils")
from models import LeNet5


# Argument Parsing

parser = argparse.ArgumentParser("ATAC-GAN MNIST")


# Config

cuda = True

n_epochs=150
batch_size=32
lr=0.0002
b1=0.5
b2=0.999
latent_dim=200
n_classes=10
img_size=28
channels=1
save_interval=500
print_interval=500
sample_interval=500

output_dir="../output/MNIST-" + str(torch.random.initial_seed())

d_real_loss_coeff = 0.7
d_fake_loss_coeff = 0.3

adv_loss_coeff = 2
aux_loss_coeff = 1
tar_loss_coeff = .08

tar_loss_default = 22.2  # This is equal to the max possible tar_loss value

# target classifier conditional constants
adv_loss_threshold = 0.8
aux_loss_threshold = 1.47


# Util

def load_LeNet5():
    net = LeNet5()
    
    # remove map location = cpu if using cuda
    net.load_state_dict(torch.load("../utils/models/trained_lenet5.pkl", map_location=torch.device('cpu')))
    
    # set model to eval mode so nothing is changed
    net.eval()
    
    return net

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    

# Models

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.target_label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, input_labels, target_labels):
        #inputs = input_labels * 10 + target_labels
        gen_input = torch.mul(self.label_emb(input_labels), noise)
        gen_input = torch.mul(self.target_label_emb(target_labels), gen_input)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 2#img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
    
if __name__ == "__main__":
    # Loss functions

    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    target_classifier_loss = nn.CrossEntropyLoss()


    # Initialize/Load Models

    generator = Generator()
    discriminator = Discriminator()

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    target_classifier = load_LeNet5()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()
        target_classifier = target_classifier.cuda()
        target_classifier_loss = target_classifier_loss.cuda()


    # Configure data loader

    os.makedirs("../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )


    # Loss helper

    def get_target_loss(adv_loss, aux_loss, target_classifier_out, target_classification):
        if (adv_loss < adv_loss_threshold and aux_loss < aux_loss_threshold):
            return target_classifier_loss(target_classification, target_classification)
        return Variable(FloatTensor([tar_loss_default]))


    # Set up logging/saving

    os.mkdir(output_dir)
    os.mkdir(output_dir + "/images")
    f = open(output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    log_writer.writerow(['Epoch', 'Batch', 'DLoss', 'DAcc', 'TarAcc', 'AdvLoss', 'AuxLoss', 'TarLoss', 'GLoss'])

    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        #labels = Variable(LongTensor(n_row ** 2).fill_(0), requires_grad=False)
        labels = Variable(LongTensor(np.array([num for _ in range(n_row) for num in range(n_row)])), requires_grad=False)
        #target_labels = Variable(LongTensor(n_row ** 2).fill_(0), requires_grad=False)
        target_labels = Variable(LongTensor(np.array([num for num in range(n_row) for _ in range(n_row)])), requires_grad=False)

        gen_imgs = generator(z, labels, target_labels)
        save_image(gen_imgs.data, output_dir + "/images/%d.png" % batches_done, nrow=n_row, normalize=True)


    # Train

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))


            # =================
            #  Train Generator
            # =================

            optimizer_G.zero_grad()

            # Create noise, input labels, and target labels for generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
            gen_target_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate images
            gen_imgs = generator(z, gen_labels, gen_target_labels)

            # Run generated images through discriminator and target classifier
            validity, pred_label = discriminator(gen_imgs)
            target_classifier_pred_label = target_classifier(gen_imgs)

            # Track target accuracy
            t_acc = np.mean(np.argmax(target_classifier_pred_label.data.cpu().numpy(), axis=1) == gen_labels.data.cpu().numpy())

            # Calculate generator loss
            adv_loss = adv_loss_coeff * adversarial_loss(validity, valid)
            aux_loss = aux_loss_coeff * auxiliary_loss(pred_label, gen_labels)
            tar_loss = tar_loss_coeff * get_target_loss(adv_loss, aux_loss, target_classifier_pred_label, gen_target_labels)
            g_loss = adv_loss + aux_loss + tar_loss

            g_loss.backward()
            optimizer_G.step()


            # =====================
            #  Train Discriminator
            # =====================

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = d_real_loss_coeff * d_real_loss + d_fake_loss_coeff * d_fake_loss

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimizer_D.step()

            log_writer.writerow([epoch, i, d_loss.item(), 100*d_acc, 100*t_acc, adv_loss.item(), aux_loss.item(), tar_loss.item(), g_loss.item()])

            batches_done = epoch * len(dataloader) + i
            if batches_done % save_interval == 0:
                # Saves weights
                torch.save(generator.state_dict(), output_dir + '/G')
                torch.save(discriminator.state_dict(), output_dir + '/D')
            if batches_done % print_interval == 0:
                print(
                    "=====================\nEpoch %d/%d, Batch %d/%d\nD loss: %f, acc: %d%% // tar acc: %d%% // adv loss: %f, aux loss: %f, tar loss: %f"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, 100 * t_acc, adv_loss.item(), aux_loss.item(), tar_loss.item())
                )
            if batches_done % sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)