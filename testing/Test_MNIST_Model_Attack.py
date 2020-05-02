import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch

# Hack to get imports, fix later
from sys import path
p = os.path.abspath("")
root_dir = "AdversarialRobustness" # Root directory of project
p = p[:p.rindex(root_dir)+len(root_dir)]
if p not in path:
    path.append(p)
import training.ATACGAN_MNIST as atacgan
from models.MNIST_Classifiers import MNIST_Classifier_Factory as Targets
from models.MNIST_Generators import MNIST_Generator_Factory as Generators
from models.MNIST_Discriminators import MNIST_Discriminator_Factory as Discriminators

parser = argparse.ArgumentParser("Test MNIST Model Attack Performance")
parser.add_argument("--model_dir", default="", type=str, help="Directory with models G and D present (Optionally provide name)")
parser.add_argument("--name", default="", type=str, help="Looks in output/{name}/ for G and D (Optionally provide model_dir)")
parser.add_argument("--test_size", type=int, default=1000, help="Number of images to be generated and filtered")

if __name__ == "__main__":
    opt = parser.parse_args()

    # Input validation
    if opt.model_dir == "" and opt.name == "":
        raise AssertionError # At least one of these values (model_dir, name) should be provided
    elif opt.model_dir != "" and opt.name != "":
        temp_dir = p+"/output/" + opt.name
        opt.model_dir = os.path.abspath(opt.model_dir)
        if opt.model_dir != temp_dir:
            raise AssertionError # Name and directory both provided but don't match
    elif opt.name != "":
        opt.model_dir=p+"/output/" + opt.name
    else:
        opt.run_name = opt.model_dir.split('/')[-1]
        opt.model_dir = os.path.abspath(opt.model_dir) 

    if (not os.path.isdir(opt.model_dir+"/results")):
        os.mkdir(opt.model_dir+"/results")

    # Obtain relevant information from training
    with open(opt.model_dir + '/constants.csv', "r") as f:
        r = csv.reader(f)
        for row in r:
            if row[0] in ["latent_dim", 'g_model', 'd_model', 't_model']:
                vars(opt)[row[0]] = row[1]

    if opt.latent_dim == None:
        print("Constants file could not be found to ensure model consistency")
        raise FileNotFoundError # Constants file not found
    else:
        opt.latent_dim = int(opt.latent_dim)

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    # Load weights
    generator = atacgan.load_generator(opt.g_model, opt.latent_dim, load_weights=True, path=opt.model_dir)
    discriminator = atacgan.load_discriminator(opt.d_model, load_weights=True, path=opt.model_dir)
    target_classifier = atacgan.load_target(opt.t_model)

    # Create noise, input labels, and target labels for generator input
    z = Variable(FloatTensor(np.random.normal(0, 1, (opt.test_size, opt.latent_dim))))
    gen_labels = Variable(LongTensor(np.random.randint(0, atacgan.n_classes, opt.test_size)), requires_grad=False)
    target_labels = Variable(LongTensor(np.random.randint(0, atacgan.n_classes, opt.test_size)), requires_grad=False)

    # Generate adversarial examples
    gen_imgs = generator(z, gen_labels, target_labels)

    # Evaluating target classifier's performance on generated examples and comparing to discriminator output
    pred_labels = target_classifier(gen_imgs)
    validity, dpred_labels = discriminator(gen_imgs)

    gen = gen_imgs.cpu().detach().numpy()
    preds = pred_labels.data.cpu().numpy()
    true = gen_labels.data.cpu().numpy()
    v = validity.data.cpu().numpy()
    dpred = dpred_labels.data.cpu().numpy()
    trlb = target_labels.data.cpu().numpy()
    indices = []
    for i in range(len(true)):
        if np.argmax(preds[i]) != true[i] and np.argmax(dpred[i]) == true[i]:
            indices.append(i)
            plt.title("Gen Label: " + str(true[i]) + "; LeNet Label: " + str(np.argmax(preds[i])))
            plt.xlabel("Discriminator Valid: " + str(v[i]))
            plt.ylabel("Target Label: " + str(trlb[i]))
            plt.imshow(gen[i][0], cmap='gray')
            plt.savefig(opt.model_dir + "/results/" + str(i) + ".png")

    # NOTE: Could potentially only take the samples with the top half validity scores

    print("%d / %d (%.1f) Adversarial" % (len(indices), opt.test_size, len(indices) * 100 / opt.test_size))
