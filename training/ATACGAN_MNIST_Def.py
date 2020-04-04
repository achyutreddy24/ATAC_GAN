cd import argparse
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
from MNIST_Generators import Generator_3Ca as Generator
from MNIST_Discriminators import Discriminator_Combined_4Ca as Discriminator


# Argument Parsing
#TBD


# Config

cuda = True

training_set_size = 60000
n_epochs=6000
batch_size=64
lr=0.0002
b1=0.5
b2=0.999
n_classes=10
latent_dim=100
img_size=28
channels=1
save_interval=1000
print_interval=200
sample_interval=2000

output_dir="../output/MNIST-" + str(torch.random.initial_seed())

d_real_adv_loss_coeff = 1
d_fake_adv_loss_coeff = 1
d_real_aux_loss_coeff = 1
d_fake_aux_loss_coeff = 1

g_adv_loss_coeff = 1
g_aux_loss_coeff = 1
g_tar_loss_coeff = 1

g_tar_loss_adv_sigm_scalar = 50
g_tar_loss_aux_sigm_scalar = 50

adv_loss_threshold = 0.71
aux_loss_threshold = 1.47

load_g = False
load_d = False
load_g_path = ""
load_d_path = ""
load_t_path = "../output/MNIST-C60000-2290918716792143116/C"


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
    o.load_state_dict(load["optimizer_state_dict"])
    m.to(d)


if __name__ == "__main__":
    # Loss functions

    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    target_classifier_loss = nn.CrossEntropyLoss()


    # Initialize Models

    generator = Generator(latent_dim)
    discriminator = Discriminator()
    target_classifier = Classifier() # Change if using different target classifier structure

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        target_classifier.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()
        target_classifier_loss.cuda()


    # Load Models

    if (load_g):
        load_model(generator, optimizer_G, load_g_path)
    else:
        generator.apply(init_weights)
    if (load_d):
        load_model(discriminator, optimizer_D, load_d_path)
    else:
        discriminator.apply(init_weights)

    target_classifier.load_state_dict(torch.load(load_t_path, map_location=torch.device('cuda' if cuda else 'cpu'))["model_state_dict"])
    target_classifier.eval()
    for param in target_classifier.parameters():
        param.requires_grad = False


    # Set max value for tar loss and c loss

    with torch.no_grad():
        pred_label = F.softmax(Variable(FloatTensor([[0, 99999, 0, 0, 0, 0, 0, 0, 0, 0]]), requires_grad=False), dim=1)
        tar_label = Variable(LongTensor([0]), requires_grad=False)
        g_tar_loss_max = target_classifier_loss(pred_label, tar_label)
        print("T labels:",pred_label)
        print("G Tar Loss Max:",g_tar_loss_max)


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


    # Set up logging/saving

    os.mkdir(output_dir)
    os.mkdir(output_dir + "/images")

    f = open(output_dir + '/constants.txt', 'a')
    f.write("cuda: {}\n".format(cuda))
    f.write("batch_size: {}\n".format(batch_size))
    f.write("lr: {}\n".format(lr))
    f.write("adam_b1: {}\n".format(b1))
    f.write("adam_b2: {}\n".format(b2))
    f.write("n_classes: {}\n".format(n_classes))
    f.write("latent_dim: {}\n".format(latent_dim))
    f.write("img_size: {}\n".format(img_size))
    f.write("channels: {}\n".format(channels))
    f.write("save_interval: {}\n".format(save_interval))
    f.write("print_interval: {}\n".format(print_interval))
    f.write("sample_interval: {}\n".format(sample_interval))
    f.write("output_dir: {}\n".format(output_dir))
    f.write("d_real_adv_loss_coeff: {}\n".format(d_real_adv_loss_coeff))
    f.write("d_real_aux_loss_coeff: {}\n".format(d_real_aux_loss_coeff))
    f.write("d_fake_adv_loss_coeff: {}\n".format(d_fake_adv_loss_coeff))
    f.write("d_real_aux_loss_coeff: {}\n".format(d_real_aux_loss_coeff))
    f.write("g_adv_loss_coeff: {}\n".format(g_adv_loss_coeff))
    f.write("g_aux_loss_coeff: {}\n".format(g_aux_loss_coeff))
    f.write("g_tar_loss_coeff: {}\n".format(g_tar_loss_coeff))
    f.write("g_tar_loss_adv_sigm_scalar: {}\n".format(g_tar_loss_adv_sigm_scalar))
    f.write("g_tar_loss_aux_sigm_scalar: {}\n".format(g_tar_loss_aux_sigm_scalar))
    f.write("g_tar_loss_max: {}\n".format(g_tar_loss_max))
    f.write("adv_loss_threshold: {}\n".format(adv_loss_threshold))
    f.write("aux_loss_threshold: {}\n".format(aux_loss_threshold))
    f.write("load_g_path: {}\n".format(load_g_path))
    f.write("load_d_path: {}\n".format(load_d_path))
    f.write("load_t_path: {}\n".format(load_t_path))
    f.close()

    f = open(output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    log_writer.writerow(['Epoch', 'Batch', 'DLoss', 'DRealLoss', 'DRealAdvLoss', 'DRealAuxLoss', 'DFakeLoss', 'DFakeAdvLoss', 'DFakeAuxLoss', 'DValidReal', 'DValidFake', 'DAccReal', 'DAccFake', 'TAcc', 'GLoss', 'GAdvLoss', 'GAuxLoss', 'GTarLossRaw', 'GTarLossWeight', 'GTarLoss'])

    def sample_image(n_row, batches_done):
        with torch.no_grad():
            # Sample noise
            z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))

            # Get labels ranging from 0 to n_classes for n rows
            labels = Variable(LongTensor(np.array([num for _ in range(n_row) for num in range(n_row)])), requires_grad=False)
            target_labels = Variable(LongTensor(np.array([num for num in range(n_row) for _ in range(n_row)])), requires_grad=False)

            # Generate images and save
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
            g_tar_loss = g_tar_loss_coeff * (g_tar_loss_max * (1 - g_tar_loss_weight) + g_tar_loss_weight * g_tar_loss_raw)

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
            d_real_loss = d_real_adv_loss_coeff * d_adv_loss_real + d_real_aux_loss_coeff * d_aux_loss_real

            # Loss for generated images
            fake_pred, fake_aux = discriminator(x.detach())
            d_adv_loss_fake = adversarial_loss(fake_pred, fake)
            d_aux_loss_fake = auxiliary_loss(fake_aux, g_labels)
            d_fake_loss = d_fake_adv_loss_coeff * d_adv_loss_fake + d_fake_aux_loss_coeff * d_aux_loss_fake

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
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
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "optimizer_state_dict": optimizer_G.state_dict(),
                    "loss": g_loss
                }, output_dir + '/G')
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": discriminator.state_dict(),
                    "optimizer_state_dict": optimizer_D.state_dict(),
                    "loss": d_loss
                }, output_dir + '/D')

            # Print information
            if batches_done % print_interval == 0:
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
