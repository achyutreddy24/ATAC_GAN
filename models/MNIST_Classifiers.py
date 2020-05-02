import torch
from torch import nn

#------------------
# Model Definitions
#------------------

class Classifier_4Ca(nn.Module):
    def __init__(self):
        super(Classifier_4Ca, self).__init__()

        def classifier_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *classifier_block(1, 16, bn=False),
            *classifier_block(16, 32),
            *classifier_block(32, 64),
            *classifier_block(64, 128),
        )

        # Output layer
        self.aux_layer = nn.Sequential(nn.Linear(128 * 2 ** 2, 10), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        label = self.aux_layer(out)

        return label

# retreived from https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/models.py
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

    
class MNIST_Classifier_Factory(object):
    #--------------------------
    # Model Selection Functions
    #--------------------------

    MODELS = {
        'LeNet5': LeNet5,
        'Classifier_4Ca': Classifier_4Ca
    }
    
    @classmethod
    def supported_models(cls):
        return cls.MODELS.keys()
    
    @classmethod
    def get_model(cls, name):
        return cls.MODELS[name]
    