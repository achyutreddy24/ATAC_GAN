"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier

from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets

from sys import path
path.append("../models")
from MNIST_Classifiers import Classifier_4Ca as Classifier

#load_path = "../output/MNIST-CuG-5356196018254729871/C"
load_path = "../output/CuG"

# Step 1: Load the MNIST dataset

transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
os.makedirs("../data/mnist", exist_ok=True)
data = datasets.MNIST(
        "../data/mnist",
        train=False,
        download=True,
        transform=transform)
testloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)

# Step 2: Create the model

# NOTE::: Maybe remove softmax from structure to improve attack effectiveness?
model = Classifier()
load = torch.load(load_path, map_location="cpu")
model.load_state_dict(load["model_state_dict"])
model.eval()
model.to("cpu")

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer.load_state_dict(load["optimizer_state_dict"])

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(-1, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)
classifier.set_learning_phase(False)

# Step 4: Evaluate the ART classifier on benign test examples

count = 0
ben_sum = 0
att_sum = 0
for i, (imgs, labels) in enumerate(testloader):
    predictions = classifier.predict(imgs)
    accuracy = np.sum(np.argmax(predictions, axis=1) == labels.numpy()) / 128
    #print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    ben_sum += accuracy

    # Step 5: Generate adversarial test examples
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    x_test_adv = attack.generate(x=imgs)

    # Step 6: Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == labels.numpy()) / 128
    #print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    att_sum += accuracy

    count += 1
print("Benign:", ben_sum/count)
print("Attack:", att_sum/count)
