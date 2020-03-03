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

cuda = True

n_epochs=500
batch_size=24
lr=0.0002
b1=0.5
b2=0.999
latent_dim=200
n_classes=10
img_size=28
channels=1
save_interval=2000
print_interval=2000
sample_interval=2000

output_dir="../output/MNIST-" + str(torch.random.initial_seed())

d_real_loss_coeff = 0.7
d_fake_loss_coeff = 0.3

g_adv_loss_coeff = 3
g_aux_loss_coeff = 1
g_tar_loss_coeff = 1

g_tar_loss_adv_sigm_scalar = 50
g_tar_loss_aux_sigm_scalar = 50

# target classifier conditional constants
adv_loss_threshold = 1.12
aux_loss_threshold = 1.475


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


    # Set max value for tar loss

    pred_label = Variable(FloatTensor([[-99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999]]), requires_grad=False)
    tar_label = Variable(LongTensor([0]), requires_grad=False)
    g_tar_loss_max = target_classifier_loss(F.softmax(pred_label, dim=1), tar_label)
    print(g_tar_loss_max)


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


    # Set up logging/saving

    os.mkdir(output_dir)
    os.mkdir(output_dir + "/images")

    f = open(output_dir + '/constants.txt', 'a')
    f.write("cuda: {}\n".format(cuda))
    f.write("batch_size: {}\n".format(batch_size))
    f.write("lr: {}\n".format(lr))
    f.write("adam_b1: {}\n".format(b1))
    f.write("adam_b2: {}\n".format(b2))
    f.write("latent_dim: {}\n".format(latent_dim))
    f.write("n_classes: {}\n".format(n_classes))
    f.write("img_size: {}\n".format(img_size))
    f.write("channels: {}\n".format(channels))
    f.write("save_interval: {}\n".format(save_interval))
    f.write("print_interval: {}\n".format(print_interval))
    f.write("sample_interval: {}\n".format(sample_interval))
    f.write("output_dir: {}\n".format(output_dir))
    f.write("d_real_loss_coeff: {}\n".format(d_real_loss_coeff))
    f.write("d_fake_loss_coeff: {}\n".format(d_fake_loss_coeff))
    f.write("g_adv_loss_coeff: {}\n".format(g_adv_loss_coeff))
    f.write("g_aux_loss_coeff: {}\n".format(g_aux_loss_coeff))
    f.write("g_tar_loss_coeff: {}\n".format(g_tar_loss_coeff))
    f.write("g_tar_loss_adv_sigm_scalar: {}\n".format(g_tar_loss_adv_sigm_scalar))
    f.write("g_tar_loss_aux_sigm_scalar: {}\n".format(g_tar_loss_aux_sigm_scalar))
    f.write("g_tar_loss_max: {}\n".format(g_tar_loss_max))
    f.write("adv_loss_threshold: {}\n".format(adv_loss_threshold))
    f.write("aux_loss_threshold: {}\n".format(aux_loss_threshold))
    f.close()

    f = open(output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    log_writer.writerow(['Epoch', 'Batch', 'DLoss', 'DRealLoss', 'DRealAdvLoss', 'DRealAuxLoss', 'DFakeLoss', 'DFakeAdvLoss', 'DFakeAuxLoss', 'DValidReal', 'DValidFake', 'DAccReal', 'DAccFake', 'TAcc', 'GLoss', 'GAdvLoss', 'GAuxLoss', 'GTarLossRaw', 'GTarLossWeight', 'GTarLoss'])

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

    running_d_real_validity = 0.0
    running_d_fake_validity = 0.0
    running_d_adv_loss_real = 0.0
    running_d_aux_loss_real = 0.0
    running_d_adv_loss_fake = 0.0
    running_d_aux_loss_fake = 0.0
    running_d_real_loss = 0.0
    running_d_fake_loss = 0.0
    running_d_loss = 0.0
    running_d_acc_real = 0.0
    running_d_acc_fake = 0.0
    running_t_acc = 0.0
    running_g_adv_loss = 0.0
    running_g_aux_loss = 0.0
    running_g_tar_loss_raw = 0.0
    running_g_tar_loss = 0.0
    running_g_tar_loss_weight = 0.0
    running_g_loss = 0.0


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
            g_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
            g_target_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate images
            x = generator(z, g_labels, g_target_labels)

            # Run generated images through discriminator and target classifier
            d_validities, d_pred_labels = discriminator(x)
            t_pred_labels = F.softmax(target_classifier(x), dim=1)

            # Generator loss components
            g_adv_loss = g_adv_loss_coeff * adversarial_loss(d_validities, valid)
            g_aux_loss = g_aux_loss_coeff * auxiliary_loss(d_pred_labels, g_labels)
            g_tar_loss_raw = target_classifier_loss(t_pred_labels, g_target_labels)
            g_tar_loss_weight = torch.sigmoid(g_tar_loss_adv_sigm_scalar * (-g_adv_loss + adv_loss_threshold)) * torch.sigmoid(g_tar_loss_aux_sigm_scalar * (-g_aux_loss + aux_loss_threshold))
            g_tar_loss = g_tar_loss_coeff * g_tar_loss_max * (1 - g_tar_loss_weight) + g_tar_loss_weight * g_tar_loss_raw

            # Total generator loss
            g_loss = g_adv_loss + g_aux_loss + g_tar_loss
            g_loss.backward()
            optimizer_G.step()


            # =====================
            #  Train Discriminator
            # =====================

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_adv_loss_real = adversarial_loss(real_pred, valid)
            d_aux_loss_real = auxiliary_loss(real_aux, labels)
            d_real_loss = (d_adv_loss_real + d_aux_loss_real) / 2

            # Loss for generated images
            fake_pred, fake_aux = discriminator(x.detach())
            d_adv_loss_fake = adversarial_loss(fake_pred, fake)
            d_aux_loss_fake = auxiliary_loss(fake_aux, g_labels)
            d_fake_loss = (d_adv_loss_fake + d_aux_loss_fake) / 2

            # Total discriminator loss
            d_loss = d_real_loss_coeff * d_real_loss + d_fake_loss_coeff * d_fake_loss
            d_loss.backward()
            optimizer_D.step()


            # =========
            #  Logging
            # =========

            # Track generator loss components
            running_g_adv_loss += g_adv_loss.item()
            running_g_aux_loss += g_aux_loss.item()
            running_g_tar_loss_raw += g_tar_loss_raw.item()
            running_g_tar_loss += g_tar_loss.item()
            running_g_tar_loss_weight += g_tar_loss_weight
            running_g_loss += g_loss.item()

            # Track target accuracy
            t_acc = np.mean(np.argmax(t_pred_labels.detach().cpu().numpy(), axis=1) == g_labels.detach().cpu().numpy())
            running_t_acc += t_acc

            # Track discriminator loss
            running_d_real_loss += d_real_loss.item()
            running_d_fake_loss += d_fake_loss.item()
            running_d_adv_loss_real += d_adv_loss_real.item()
            running_d_aux_loss_real += d_aux_loss_real.item()
            running_d_adv_loss_fake += d_adv_loss_fake.item()
            running_d_aux_loss_fake += d_aux_loss_fake.item()
            running_d_loss += d_loss.item()

            # Track dicriminator output
            d_validity_fake = np.mean(d_validities.detach().cpu().numpy())
            running_d_fake_validity += d_validity_fake
            d_validity_real = np.mean(real_pred.detach().cpu().numpy())
            running_d_real_validity += d_validity_real

            # Track discriminator-classifier accuracy
            d_acc_real = np.mean(np.argmax(real_aux.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy())
            d_acc_fake = np.mean(np.argmax(fake_aux.detach().cpu().numpy(), axis=1) == g_labels.detach().cpu().numpy())
            running_d_acc_real += d_acc_real
            running_d_acc_fake += d_acc_fake

            # Write to log file
            log_writer.writerow([epoch, i, d_loss.item(), d_real_loss.item(), d_adv_loss_real.item(), d_aux_loss_real.item(), d_fake_loss.item(), d_adv_loss_fake.item(), d_aux_loss_fake.item(), d_validity_real, d_validity_fake, 100*d_acc_real, 100*d_acc_fake, 100*t_acc, g_loss.item(), g_adv_loss.item(), g_aux_loss.item(), g_tar_loss_raw.item(), g_tar_loss_weight.item(), g_tar_loss.item()])


            batches_done = epoch * len(dataloader) + i

            # Save models
            if batches_done % save_interval == 0:
                # Saves weights
                torch.save(generator.state_dict(), output_dir + '/G')
                torch.save(discriminator.state_dict(), output_dir + '/D')

            # Print information
            if batches_done % print_interval == 0:
                #TEMP
                #print("d_acc_real", d_acc_real)
                #print("d_acc_fake", d_acc_fake)
                #END

                p = float(print_interval)
                print("==============================")
                print("Epoch %d/%d, Batch %d/%d" % (epoch, n_epochs, i, len(dataloader)))
                print("D - Real Valid: %f, Fake Valid: %f, Real Acc: %.2f%%, Fake Acc: %.2f%%" % (running_d_real_validity / p, running_d_fake_validity / p, running_d_acc_real * 100 / p, running_d_acc_fake * 100 / p))
                print("D Loss: %f" % (running_d_loss / p))
                print("   Real Loss: %f  (Adv: %f, Aux: %f)" % (running_d_real_loss / p, running_d_adv_loss_real / p, running_d_aux_loss_real / p))
                print("   Fake Loss: %f  (Adv %f, Aux: %f)" % (running_d_fake_loss / p, running_d_adv_loss_fake / print_interval, running_d_aux_loss_fake / p))
                print("G Loss: %f (Adv: %f, Aux: %f, Tar: %f)" % (running_g_loss / p, running_g_adv_loss / p, running_g_aux_loss  / p, running_g_tar_loss / p))
                print("   Tar Raw: %f, Tar Weight: %f" % (running_g_tar_loss_raw / p, running_g_tar_loss_weight / p))
                print("Tar Acc: %.2f%%" % (running_t_acc * 100 / p))

                running_d_real_validity = 0.0
                running_d_fake_validity = 0.0
                running_d_adv_loss_real = 0.0
                running_d_aux_loss_real = 0.0
                running_d_adv_loss_fake = 0.0
                running_d_aux_loss_fake = 0.0
                running_d_real_loss = 0.0
                running_d_fake_loss = 0.0
                running_d_loss = 0.0
                running_d_acc_real = 0.0
                running_d_acc_fake = 0.0
                running_t_acc = 0.0
                running_g_adv_loss = 0.0
                running_g_aux_loss = 0.0
                running_g_tar_loss_raw = 0.0
                running_g_tar_loss = 0.0
                running_g_tar_loss_weight = 0.0
                running_g_loss = 0.0

            # Save sample images
            if batches_done % sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)
