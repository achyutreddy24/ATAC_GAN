import torch
from torch import nn

class netG(nn.Module):
    # https://github.com/clvrai/ACGAN-PyTorch/blob/master/network.py Cifar10 model
    # Modified to include label and target embeddings
    def __init__(self, nz, ngpu=1):
        super(netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        
        self.label_emb = nn.Embedding(10, nz)
        self.target_label_emb = nn.Embedding(10, nz)

        # first linear layer
        self.fc1 = nn.Linear(nz, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, noise, input_labels, target_labels):
        noise = noise.view(-1, self.nz)
        gen_input = torch.mul(self.label_emb(input_labels), noise)
        gen_input = torch.mul(self.target_label_emb(target_labels), gen_input)
        fc1 = self.fc1(gen_input)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5
        return output

class Generator_3Ca(nn.Module):
    def __init__(self, latent_dim):
        super(Generator_3Ca, self).__init__()
        
        self.label_emb = nn.Embedding(10, latent_dim)
        self.target_label_emb = nn.Embedding(10, latent_dim)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * 7 ** 2))

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
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, input_labels, target_labels):
        gen_input = torch.mul(self.label_emb(input_labels), noise)
        gen_input = torch.mul(self.target_label_emb(target_labels), gen_input)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, 7, 7)
        img = self.conv_blocks(out)
        return img
    
class Cifar10_Generator_Factory(object):
    #--------------------------
    # Model Selection Functions
    #--------------------------

    MODELS = {
        'netG': netG,
    }
    
    @classmethod
    def supported_models(cls):
        return cls.MODELS.keys()
    
    @classmethod
    def get_model(cls, name):
        return cls.MODELS[name]