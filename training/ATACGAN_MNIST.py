import argparse
import os
import numpy as np
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
# Hack to get imports, fix later
from sys import path
p = os.path.abspath("")
root_dir = "AdversarialRobustness" # Root directory of project
p = p[:p.rindex(root_dir)+len(root_dir)]
if p not in path:
    path.append(p)
#from models.MNIST_Classifiers import LeNet5 as Target
#from models.MNIST_Generators import Generator_3Ca as Generator
#from models.MNIST_Discriminators import Discriminator_Combined_4Ca as Discriminator
from models.MNIST_Classifiers import MNIST_Classifier_Factory as Targets
from models.MNIST_Generators import MNIST_Generator_Factory as Generators
from models.MNIST_Discriminators import MNIST_Discriminator_Factory as Discriminators

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of latent space")
parser.add_argument("--output_dir", type=str, default="", help="Leave blank to auto-generate under /output")

parser.add_argument("--name", type=str, default="", help="Leave blank to auto-generate name")

parser.add_argument("--d_real_adv_loss_coeff", type=float, default=0.3)
parser.add_argument("--d_real_aux_loss_coeff", type=float, default=0.3)
parser.add_argument("--d_fake_adv_loss_coeff", type=float, default=0.3)
parser.add_argument("--d_fake_aux_loss_coeff", type=float, default=0.12)

parser.add_argument("--g_adv_loss_coeff", type=float, default=3.0)
parser.add_argument("--g_aux_loss_coeff", type=float, default=1.0)
parser.add_argument("--g_tar_loss_coeff", type=float, default=1.0)

parser.add_argument("--g_tar_loss_adv_sigm_scalar", type=int, default=50)
parser.add_argument("--g_tar_loss_aux_sigm_scalar", type=int, default=50)

# target classifier conditional constants
parser.add_argument("--adv_loss_threshold", type=float, default=2.13)
parser.add_argument("--aux_loss_threshold", type=float, default=1.48)

parser.add_argument("--resume", type=bool, default=False, help="Resumes training from output_dir")

parser.add_argument("--save_interval", type=int, default=2000, help="How frequent model weights are saved")
parser.add_argument("--print_interval", type=int, default=2000, help="Print frequency, -1 to disable")
parser.add_argument("--sample_interval", type=int, default=2000, help="Image save frequency, -1 to disable")

parser.add_argument("--tb", default=False, action="store_true", help="Enable tensorboard logging under output/runs/name")

# Model Choices
parser.add_argument(
    '--t_model',
    type=str,
    help='Name of model to train',
    choices=Targets.supported_models(),
    required=True,
)
# Model Choices
parser.add_argument(
    '--d_model',
    type=str,
    help='Name of model to train',
    choices=Discriminators.supported_models(),
    required=True,
)
# Model Choices
parser.add_argument(
    '--g_model',
    type=str,
    help='Name of model to train',
    choices=Generators.supported_models(),
    required=True,
)

# MNIST Constants
n_classes=10 # Number of classes for dataset
img_size=28 # Size of each image dimension
channels=1 # Number of image channels

# Util

def load_target(name):
    net = Targets.get_model(name)()
    net.load_state_dict(torch.load(p+"/models/saves/MNIST_LeNet5")) # Generalize this

    # set model to eval mode so nothing is changed
    net.eval()
    return net

def load_discriminator(name, load_weights=False, path=None):
    net = Discriminators.get_model(name)()
    if load_weights and path != None:
        net.load_state_dict(torch.load(path+"/D"))
    return net

def load_generator(name, latent_dim, load_weights=False, path=None):
    net = Generators.get_model(name)(latent_dim)
    if load_weights and path != None:
        net.load_state_dict(torch.load(path+"/G"))
    return net

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    
    opt = parser.parse_args()

    # Checking if this model can be safely resumed
    if opt.resume:
        if opt.output_dir == "":
            raise AssertionError # Nothing to resume
        # Loads constants from previous run
        r = csv.reader(open(opt.output_dir + '/constants.csv', "r"))
        for row in r:
            if row[0] == "cuda":
                continue
            elif row[0] in ['g_model', 'd_model', 't_model']:
                if vars(opt)[row[0]] != row[1]:
                    raise AssertionError # Training was resumed with inconsistent model architecture
            else:
                try:
                    vars(opt)[row[0]] = int(float(row[1])) if int(float(row[1])) == float(row[1]) else float(row[1])
                except ValueError:
                    vars(opt)[row[0]] = row[1]


    # Setting run name and correct output directory for all possible inputs
    if opt.output_dir == "" and opt.name == "":
        opt.name = "MNIST-" + str(torch.random.initial_seed())
        opt.output_dir=p+"/output/" + opt.name
    elif opt.output_dir == "":
        opt.output_dir=p+"/output/" + opt.name
    elif opt.name == "":
        opt.name = opt.output_dir.split('/')[-1]

    # Run on GPU if available
    cuda = True if torch.cuda.is_available() else False
    opt.cuda = cuda # For logging

    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    target_classifier_loss = nn.CrossEntropyLoss()


    # Loading models (will load previous weights if resuming)
    generator = load_generator(opt.g_model, opt.latent_dim, opt.resume, path=opt.output_dir)
    discriminator = load_discriminator(opt.d_model, opt.resume, path=opt.output_dir)
    target_classifier = load_target(opt.t_model)

    # Initializing weights
    if opt.resume == False:
        generator.apply(init_weights)
        discriminator.apply(init_weights)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()
        target_classifier.cuda()
        target_classifier_loss.cuda()


    # Set max value for tar loss
    
    pred_label = Variable(FloatTensor([[-99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999]]), requires_grad=False)
    tar_label = Variable(LongTensor([0]), requires_grad=False)
    g_tar_loss_max = target_classifier_loss(F.softmax(pred_label, dim=1), tar_label)

    
    # Configure data loader
    
    os.makedirs(p+"/data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            p+"/data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Set up logging/saving
    
    if opt.resume == False:
        os.mkdir(opt.output_dir)
        os.mkdir(opt.output_dir + "/images")
        
    # Setup tensorboard logging
    if opt.tb:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(p+"/output/runs/"+opt.name)
        #tb_writer.add_graph((generator))

    # Save command line constants to file
    with open(opt.output_dir + '/constants.csv', "w") as f:
        w = csv.writer(f)
        for key, val in vars(opt).items():
            if key in ['output_dir', 'resume']:
                continue
            w.writerow([key, val])

    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        labels = Variable(LongTensor(np.array([num for _ in range(n_row) for num in range(n_row)])), requires_grad=False)
        target_labels = Variable(LongTensor(np.array([num for num in range(n_row) for _ in range(n_row)])), requires_grad=False)
        gen_imgs = generator(z, labels, target_labels)
        save_image(gen_imgs.data, opt.output_dir + "/images/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Only used for print messages
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
    
    # Starting log
    f = open(opt.output_dir + '/log.csv', 'a')
    log_writer = csv.writer(f, delimiter=',')
    if opt.resume == False: 
        # Log file header
        log_writer.writerow(['Epoch', 'Batch', 'DLoss', 'DRealLoss', 'DRealAdvLoss', 'DRealAuxLoss', 'DFakeLoss', 'DFakeAdvLoss', 'DFakeAuxLoss', 'DValidReal', 'DValidFake', 'DAccReal', 'DAccFake', 'TAcc', 'GLoss', 'GAdvLoss', 'GAuxLoss', 'GTarLossRaw', 'GTarLossWeight', 'GTarLoss'])

    # Find starting point for training
    start_epoch = 0
    start_batch = 0 # Unimplemented but probably not important
    # Finish the number of epochs the run was originally set to by using logged values
    if opt.resume:
        import subprocess
        tail = subprocess.Popen(['tail', '-n', '1', opt.output_dir + '/log.csv'], stdout=subprocess.PIPE)
        last_line = tail.stdout.readline().decode().split(',')
        start_epoch = int(last_line[0]) # Epoch last run ended on
        start_batch = int(last_line[1]) # Epoch last run ended on


    # Training
    print("Started training model in", opt.output_dir)
    
    for epoch in range(start_epoch, opt.n_epochs):

        for i, (imgs, labels) in enumerate(dataloader):
            
            opt.batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            
            # =================
            #  Train Generator
            # =================

            optimizer_G.zero_grad()

            # Create noise, input labels, and target labels for generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
            g_labels = Variable(LongTensor(np.random.randint(0, n_classes, opt.batch_size)))
            g_target_labels = Variable(LongTensor(np.random.randint(0, n_classes, opt.batch_size)))

            # Generate images
            x = generator(z, g_labels, g_target_labels)

            # Run generated images through discriminator and target classifier
            d_validities, d_pred_labels = discriminator(x)
            t_pred_labels = F.softmax(target_classifier(x), dim=1)

            # Generator loss components
            g_adv_loss = opt.g_adv_loss_coeff * adversarial_loss(d_validities, valid)
            g_aux_loss = opt.g_aux_loss_coeff * auxiliary_loss(d_pred_labels, g_labels)
            g_tar_loss_raw = target_classifier_loss(t_pred_labels, g_target_labels)
            g_tar_loss_weight = torch.sigmoid(opt.g_tar_loss_adv_sigm_scalar * (-g_adv_loss + opt.adv_loss_threshold)) * torch.sigmoid(opt.g_tar_loss_aux_sigm_scalar * (-g_aux_loss + opt.aux_loss_threshold))
            g_tar_loss = opt.g_tar_loss_coeff * g_tar_loss_max * (1 - g_tar_loss_weight) + g_tar_loss_weight * g_tar_loss_raw

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
            d_real_loss = opt.d_real_adv_loss_coeff * d_adv_loss_real + opt.d_real_aux_loss_coeff * d_aux_loss_real

            # Loss for generated images
            fake_pred, fake_aux = discriminator(x.detach())
            d_adv_loss_fake = adversarial_loss(fake_pred, fake)
            d_aux_loss_fake = auxiliary_loss(fake_aux, g_labels)
            d_fake_loss = opt.d_fake_adv_loss_coeff * d_adv_loss_fake + opt.d_fake_aux_loss_coeff * d_aux_loss_fake

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            
            # =========
            #  Logging
            # =========
            
            # Track target accuracy
            t_acc = np.mean(np.argmax(t_pred_labels.detach().cpu().numpy(), axis=1) == g_labels.detach().cpu().numpy())
            
            # Track discriminator-classifier accuracy
            d_acc_real = np.mean(np.argmax(real_aux.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy())
            d_acc_fake = np.mean(np.argmax(fake_aux.detach().cpu().numpy(), axis=1) == g_labels.detach().cpu().numpy())
                        
            # Track discriminator output
            d_validity_fake = np.mean(d_validities.detach().cpu().numpy())
            d_validity_real = np.mean(real_pred.detach().cpu().numpy())

            # Write all loss components to log file
            log_writer.writerow([epoch, i, d_loss.item(), d_real_loss.item(), d_adv_loss_real.item(), d_aux_loss_real.item(), d_fake_loss.item(), d_adv_loss_fake.item(), d_aux_loss_fake.item(), d_validity_real, d_validity_fake, 100*d_acc_real, 100*d_acc_fake, 100*t_acc, g_loss.item(), g_adv_loss.item(), g_aux_loss.item(), g_tar_loss_raw.item(), g_tar_loss_weight.item(), g_tar_loss.item()])

            # Total number of batches over all epochs
            batches_done = epoch * len(dataloader) + i
            
            # Write loss components to tensorboard directory
            if opt.tb:
                tb_writer.add_scalars('G-D loss', {'d_loss': d_loss, 'g_loss': g_loss}, batches_done)
                tb_writer.add_scalars('G loss components', {'adv_loss': g_adv_loss, 'g_aux_loss': g_aux_loss, 'g_tar_loss': g_tar_loss}, batches_done)

            # Save model weights
            if batches_done % opt.save_interval == 0:
                torch.save(generator.state_dict(), opt.output_dir + '/G')
                torch.save(discriminator.state_dict(), opt.output_dir + '/D')
                
            # Save sample images
            if opt.sample_interval != 0 and batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)

            # Print information only when print interval is not -1
            if opt.print_interval != -1:
                # Running discriminator-classifier accuracy
                running_d_acc_real += d_acc_real
                running_d_acc_fake += d_acc_fake

                # Running target accuracy
                running_t_acc += t_acc

                # Running discriminator output
                running_d_fake_validity += d_validity_fake
                running_d_real_validity += d_validity_real

                # Running discriminator loss
                running_d_real_loss += d_real_loss.item()
                running_d_fake_loss += d_fake_loss.item()
                running_d_adv_loss_real += d_adv_loss_real.item()
                running_d_aux_loss_real += d_aux_loss_real.item()
                running_d_adv_loss_fake += d_adv_loss_fake.item()
                running_d_aux_loss_fake += d_aux_loss_fake.item()
                running_d_loss += d_loss.item()

                # Running generator loss components
                running_g_adv_loss += g_adv_loss.item()
                running_g_aux_loss += g_aux_loss.item()
                running_g_tar_loss_raw += g_tar_loss_raw.item()
                running_g_tar_loss += g_tar_loss.item()
                running_g_tar_loss_weight += g_tar_loss_weight
                running_g_loss += g_loss.item()
                
                # Print these running totals every print interval
                if batches_done % opt.print_interval == 0:
                    p = float(opt.print_interval)
                    print("==============================")
                    print("Epoch %d/%d, Batch %d/%d" % (epoch, opt.n_epochs, i, len(dataloader)))
                    print("D - Real Valid: %f, Fake Valid: %f, Real Acc: %.2f%%, Fake Acc: %.2f%%" % (running_d_real_validity / p, running_d_fake_validity / p, running_d_acc_real * 100 / p, running_d_acc_fake * 100 / p))
                    print("D Loss: %f" % (running_d_loss / p))
                    print("   Real Loss: %f  (Adv: %f, Aux: %f)" % (running_d_real_loss / p, running_d_adv_loss_real / p, running_d_aux_loss_real / p))
                    print("   Fake Loss: %f  (Adv %f, Aux: %f)" % (running_d_fake_loss / p, running_d_adv_loss_fake / p, running_d_aux_loss_fake / p))
                    print("G Loss: %f (Adv: %f, Aux: %f, Tar: %f)" % (running_g_loss / p, running_g_adv_loss / p, running_g_aux_loss  / p, running_g_tar_loss / p))
                    print("   Tar Raw: %f, Tar Weight: %f" % (running_g_tar_loss_raw / p, running_g_tar_loss_weight / p))
                    print("Tar Acc: %.2f%%" % (running_t_acc * 100 / p))

                    # Set running totals to 0 for next print interval
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
    
    print("Finished training model in", opt.output_dir)
    torch.save(generator.state_dict(), opt.output_dir + '/G')
    torch.save(discriminator.state_dict(), opt.output_dir + '/D')
