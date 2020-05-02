import torch
from torch import nn

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
    
class MNIST_Generator_Factory(object):
    #--------------------------
    # Model Selection Functions
    #--------------------------

    MODELS = {
        'Generator_3Ca': Generator_3Ca,
    }
    
    @classmethod
    def supported_models(cls):
        return cls.MODELS.keys()
    
    @classmethod
    def get_model(cls, name):
        return cls.MODELS[name]