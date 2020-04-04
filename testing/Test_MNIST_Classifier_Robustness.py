"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist

from torchvision.utils import save_image

from sys import path
path.append("../models")
from MNIST_Classifiers import Classifier_4Ca as Classifier

#load_path = "../output/MNIST-CuG-5356196018254729871/C"
load_path = "../output/MNIST-C60000-2290918716792143116/C"

# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 2, 3).astype(np.float32)
x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_train = np.swapaxes(x_train, 2, 3).astype(np.float32)

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
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)
classifier.set_learning_phase(False)

# TEMP

i = 0
for (img, label) in zip(x_test, y_test):
    save_image(torch.FloatTensor(img), "images/" + str(i) + "-" + str(np.argmax(label)) + ".png", nrow=1, normalize=True)
    i+=1
    if (i > 100):
        break

# Step 4: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
'''
print(predictions[0])
print(y_test[0])
print()
print(predictions[1])
print(y_test[1])
print()
print(predictions[2])
print(y_test[2])
print()
print(predictions[3])
print(y_test[3])
print()
print(predictions[4])
print(y_test[4])
print()
'''
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 5: Generate adversarial test examples
attack = FastGradientMethod(classifier=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 6: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
