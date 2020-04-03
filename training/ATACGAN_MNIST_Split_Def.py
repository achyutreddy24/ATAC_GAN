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
path.append("../models")
from MNIST_Classifiers import Classifier_4Ca as Classifier
from MNIST_Generators import Generator_3Ca as Generator
from MNIST_Discriminators import Discriminator_4Ca as Discriminator


# Argument Parsing
#TBD


# Config

cuda = True

training_set_size = 30000
n_epochs=6000
batch_size=64
lr=0.0002
b1=0.5
b2=0.999
latent_dim=100
n_classes=10
img_size=28
channels=1
save_interval=500
print_interval=200
sample_interval=1000
test_interval=400

output_dir="../output/MNIST-" + str(torch.random.initial_seed())

d_real_loss_coeff = 0.5
d_fake_loss_coeff = 0.5

c_real_loss_coeff = 0.7
c_fake_loss_coeff = 0.3

g_adv_loss_coeff = 1
g_aux_loss_coeff = 1
g_tar_loss_coeff = 1

g_tar_loss_adv_sigm_scalar = 50
g_tar_loss_aux_sigm_scalar = 50

adv_loss_threshold = 0.71
aux_loss_threshold = 1.475

c_fake_loss_threshold = 2
c_fake_sigm_scalar = 8

load_g = False
load_d = False
load_c = True
load_g_path = ""
load_d_path = ""
load_c_path = "../output/MNIST-C-2413946350108299530/C"


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
    classifier = Classifier()
    target_classifier = Classifier() # Change if using different target classifier structure
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(b1, b2))
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        classifier.cuda()
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
    if (load_c):
        load_model(classifier, optimizer_C, load_c_path)
    else:
        classifier.apply(init_weights)

    target_classifier.load_state_dict(torch.load(load_c_path, map_location=torch.device('cuda' if cuda else 'cpu'))["model_state_dict"])
    target_classifier.eval()
    for param in target_classifier.parameters():
        param.requires_grad = False


    # Set max value for tar loss and c loss

    with torch.no_grad():
        pred_label = Variable(FloatTensor([[-99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999]]), requires_grad=False)
        tar_label = Variable(LongTensor([0]), requires_grad=False)
        g_tar_loss_max = target_classifier_loss(F.softmax(pred_label, dim=1), tar_label)
        c_fake_loss_max = auxiliary_loss(F.softmax(pred_label, dim=1), tar_label)
        
    
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
    f.write("c_real_loss_coeff: {}\n".format(c_real_loss_coeff))
    f.write("c_fake_loss_coeff: {}\n".format(c_fake_loss_coeff))
    f.write("g_adv_loss_coeff: {}\n".format(g_adv_loss_coeff))
    f.write("g_aux_loss_coeff: {}\n".format(g_aux_loss_coeff))
    f.write("g_tar_loss_coeff: {}\n".format(g_tar_loss_coeff))
    f.write("g_tar_loss_adv_sigm_scalar: {}\n".format(g_tar_loss_adv_sigm_scalar))
    f.write("g_tar_loss_aux_sigm_scalar: {}\n".format(g_tar_loss_aux_sigm_scalar))
    f.write("g_tar_loss_max: {}\n".format(g_tar_loss_max))
    f.write("adv_loss_threshold: {}\n".format(adv_loss_threshold))
    f.write("aux_loss_threshold: {}\n".format(aux_loss_threshold))
    f.write("load_c: {}\n".format(load_c))
    f.write("load_g_path: {}\n".format(load_g_path))
    f.write("load_d_path: {}\n".format(load_d_path))
    f.write("load_c_path: {}\n".format(load_c_path))
    f.close()

    f = open(output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    log_writer.writerow(['Epoch', 'Batch', 'DLoss', 'DRealLoss', 'DFakeLoss', 'DValidReal', 'DValidFake', 'CLoss', 'CRealLoss', 'CFakeLoss', 'CAccReal', 'CAccFake', 'TAcc', 'GLoss', 'GAdvLoss', 'GAuxLoss', 'GTarLossRaw', 'GTarLossWeight', 'GTarLoss'])

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
    running_d_real_loss = 0.0
    running_d_fake_loss = 0.0
    running_d_loss = 0.0
    running_c_real_loss = 0.0
    running_c_fake_loss = 0.0
    running_c_fake_loss_raw = 0.0
    running_c_fake_loss_weight = 0.0
    running_c_loss = 0.0
    running_c_acc_real = 0.0
    running_c_acc_fake = 0.0
    running_t_acc = 0.0
    running_g_adv_loss = 0.0
    running_g_aux_loss = 0.0
    running_g_tar_loss_raw = 0.0
    running_g_tar_loss = 0.0
    running_g_tar_loss_weight = 0.0
    running_g_loss = 0.0
    
    
    # Set up testing
    
    def test_c():
        classifier.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images = Variable(images.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))
                out = classifier(images)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of classifier on test set: %.3f%%' % (100 * correct / total))
        classifier.train()


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
            d_validities = discriminator(x)
            c_pred_labels = F.softmax(classifier(x), dim=1)
            t_pred_labels = F.softmax(target_classifier(x), dim=1)

            # Generator loss components
            g_adv_loss = adversarial_loss(d_validities, valid)
            g_aux_loss = auxiliary_loss(c_pred_labels, g_labels)
            g_tar_loss_raw = target_classifier_loss(t_pred_labels, g_target_labels)
            g_tar_loss_weight = torch.sigmoid(g_tar_loss_adv_sigm_scalar * (-g_adv_loss + adv_loss_threshold)) * torch.sigmoid(g_tar_loss_aux_sigm_scalar * (-g_aux_loss + aux_loss_threshold))
            g_tar_loss = g_tar_loss_max * (1 - g_tar_loss_weight) + g_tar_loss_weight * g_tar_loss_raw

            # Total generator loss
            g_loss = g_adv_loss_coeff * g_adv_loss + g_aux_loss_coeff * g_aux_loss + g_tar_loss_coeff * g_tar_loss
            g_loss.backward()
            optimizer_G.step()


            # =====================
            #  Train Discriminator
            # =====================

            optimizer_D.zero_grad()

            # Loss for real images
            d_real_pred = discriminator(real_imgs)
            d_loss_real = adversarial_loss(d_real_pred, valid)

            # Loss for generated images
            d_fake_pred = discriminator(x.detach())
            d_loss_fake = adversarial_loss(d_fake_pred, fake)

            # Total discriminator loss
            d_loss = d_real_loss_coeff * d_loss_real + d_fake_loss_coeff * d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # ============================
            #  Train Auxiliary Classifier
            # ============================
            
            optimizer_C.zero_grad()
            
            # Loss for real images
            c_real_pred = F.softmax(classifier(real_imgs), dim=1)
            c_loss_real = auxiliary_loss(c_real_pred, labels)
            
            # Loss for fake images
            c_fake_pred = F.softmax(classifier(x.detach()), dim=1)
            c_loss_fake_raw = auxiliary_loss(c_fake_pred, g_labels)
            c_fake_loss_weight = torch.sigmoid(c_fake_sigm_scalar * (-c_loss_fake_raw + c_fake_loss_threshold))
            c_loss_fake = c_fake_loss_weight * c_loss_fake_raw + (1 - c_fake_loss_weight) * c_fake_loss_max
            
            # Total auxiliary classifier loss
            c_loss = c_real_loss_coeff * c_loss_real + c_fake_loss_coeff * c_loss_fake
            c_loss.backward()
            optimizer_C.step()

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
            t_acc = np.mean(np.argmax(t_pred_labels.detach().cpu().numpy(), axis=1) == g_labels.detach().cpu().numpy()) * 100
            running_t_acc += t_acc

            # Track discriminator loss
            running_d_real_loss += d_loss_real.item()
            running_d_fake_loss += d_loss_fake.item()
            running_d_loss += d_loss.item()
            
            # Track auxiliary classifier loss
            running_c_real_loss += c_loss_real.item()
            running_c_fake_loss += c_loss_fake.item()
            running_c_loss += c_loss.item()
            running_c_fake_loss_raw += c_loss_fake_raw.item()
            running_c_fake_loss_weight += c_fake_loss_weight

            # Track dicriminator output
            d_validity_fake = np.mean(d_fake_pred.detach().cpu().numpy())
            running_d_fake_validity += d_validity_fake
            d_validity_real = np.mean(d_real_pred.detach().cpu().numpy())
            running_d_real_validity += d_validity_real

            # Track discriminator-classifier accuracy
            c_acc_real = np.mean(np.argmax(c_real_pred.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy()) * 100
            c_acc_fake = np.mean(np.argmax(c_fake_pred.detach().cpu().numpy(), axis=1) == g_labels.detach().cpu().numpy()) * 100
            running_c_acc_real += c_acc_real
            running_c_acc_fake += c_acc_fake

            # Write to log file
            log_writer.writerow([epoch, i, d_loss.item(), d_loss_real.item(), d_loss_fake.item(), d_validity_real, d_validity_fake, c_loss, c_loss_real, c_loss_fake, c_acc_real, c_acc_fake, t_acc, g_loss.item(), g_adv_loss.item(), g_aux_loss.item(), g_tar_loss_raw.item(), g_tar_loss_weight.item(), g_tar_loss.item()])

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
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer_C.state_dict(),
                    "loss": c_loss
                }, output_dir + '/C')

            # Print information
            if batches_done % print_interval == 0:
                #TEMP
                #print("d_acc_real", d_acc_real)
                #print("d_acc_fake", d_acc_fake)
                #END

                p = float(print_interval)
                print("==============================")
                print("Epoch %d/%d, Batch %d/%d" % (epoch, n_epochs, i, len(dataloader)))
                print("D Loss: %.4f (Real: %.4f, Fake: %.4f, Real Validity: %.3f, Fake Validity: %.3f)" % (running_d_loss / p, running_d_real_loss / p, running_d_fake_loss / p, running_d_real_validity / p, running_d_fake_validity / p))
                print("C Loss: %.4f (Real: %.4f, Fake: %.4f, Real Acc: %.2f%%, Fake Acc: %.2f%%)" % (running_c_loss / p, running_c_real_loss / p, running_c_fake_loss / p, running_c_acc_real / p, running_c_acc_fake / p))
                print("   Fake Raw: %.5f, Fake Weight: %.5f" % (running_c_fake_loss_raw / p, running_c_fake_loss_weight / p))
                print("G Loss: %.4f (Adv: %.4f, Aux: %.4f, Tar: %.4f)" % (running_g_loss / p, running_g_adv_loss / p, running_g_aux_loss  / p, running_g_tar_loss / p))
                print("   Tar Raw: %.5f, Tar Weight: %.5f" % (running_g_tar_loss_raw / p, running_g_tar_loss_weight / p))
                print("Tar Acc: %.2f%%" % (running_t_acc / p))

                running_d_real_validity = 0.0
                running_d_fake_validity = 0.0
                running_d_real_loss = 0.0
                running_d_fake_loss = 0.0
                running_d_loss = 0.0
                running_c_real_loss = 0.0
                running_c_fake_loss = 0.0
                running_c_fake_loss_raw = 0.0
                running_c_fake_loss_weight = 0.0
                running_c_loss = 0.0
                running_c_acc_real = 0.0
                running_c_acc_fake = 0.0
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
            
            # Test classifier
            if batches_done % test_interval == 0:
                test_c()
