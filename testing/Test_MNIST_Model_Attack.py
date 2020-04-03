from sys import path
import os
path.append("../training")
import ATACGAN_MNIST as atacgan
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("Test MNIST Model Attack Performance")
parser.add_argument("--model_dir", type=str, help="Directory with models G and D present")
parser.add_argument("--test_size", type=int, default=1000, help="Number of images to be generated and filtered")
args = parser.parse_args()

if (args.model_dir[len(args.model_dir)-1] != '/'):
    args.model_dir += '/'
if (not os.path.isdir(args.model_dir+"results")):
    os.mkdir(args.model_dir+"results")

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

generator = atacgan.Generator()
generator.load_state_dict(torch.load(args.model_dir + "G", map_location=torch.device('cpu')))
discriminator = atacgan.Discriminator()
discriminator.load_state_dict(torch.load(args.model_dir + "D", map_location=torch.device('cpu')))

z = Variable(FloatTensor(np.random.normal(0, 1, (args.test_size, atacgan.latent_dim))))
gen_labels = Variable(LongTensor(np.random.randint(0, atacgan.n_classes, args.test_size)), requires_grad=False)
target_labels = Variable(LongTensor(np.random.randint(0, atacgan.n_classes, args.test_size)), requires_grad=False)

gen_imgs = generator(z, gen_labels, target_labels)
target_classifier = atacgan.load_LeNet5().cpu()
pred_labels = target_classifier(gen_imgs)
validity, dpred_labels = discriminator(gen_imgs)

gen = gen_imgs.cpu().detach().numpy()
preds = pred_labels.data.cpu().numpy()
true = gen_labels.data.cpu().numpy()
v = validity.data.cpu().numpy()
dpred = dpred_labels.data.cpu().numpy()
indices = []
for i in range(len(true)):
    if np.argmax(preds[i]) != true[i] and np.argmax(dpred[i]) == true[i]:
        indices.append(i)
        plt.title("Gen Label: " + str(true[i]) + "; LeNet Label: " + str(np.argmax(preds[i])))
        plt.xlabel("Discriminator Valid: " + str(v[i]))
        plt.imshow(gen[i][0], cmap='gray')
        plt.savefig(args.model_dir + "results/" + str(i) + ".png")

# NOTE: Could potentially only take the samples with the top half validity scores

print("%d / %d (%.1f) Adversarial" % (len(indices), args.test_size, len(indices) * 100 / args.test_size))
