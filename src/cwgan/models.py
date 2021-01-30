# Define generator and discriminator
# References:
#   https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
#   https://github.com/u7javed/Conditional-WGAN-GP/blob/master/models.py
#   https://github.com/gcucurull/cond-wgan-gp/blob/master/models.py
import numpy as np
import torch
import torch.nn as nn


def block(in_features, out_features, normalize=True, dropout_p=None):
    layers = [nn.Linear(in_features, out_features)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_features, 0.8))
    if dropout_p is not None:
        layers.append(nn.Dropout(dropout_p))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(self, img_shape, latent_size, num_classes, embedding_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            *block(latent_size + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, noises, labels):
        # concatenate latent vector (noise) and embedded label
        gen_input = torch.cat((self.label_embedding(labels), noises), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes, embedding_dim, sigmoid=False):
        """[summary]

        Args:
            img_shape ([type]): [description]
            num_classes ([type]): [description]
            embedding_dim ([type]): [description]
            sigmoid (bool, optional): [description]. Defaults to False.
        """
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            *block(embedding_dim + int(np.prod(img_shape)),
                   512, normalize=False),
            *block(512, 512, normalize=False, dropout_p=0.4),
            *block(512, 512, normalize=False, dropout_p=0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ) if sigmoid else nn.Sequential(
            *block(embedding_dim + int(np.prod(img_shape)),
                   512, normalize=False),
            *block(512, 512, normalize=False, dropout_p=0.4),
            *block(512, 512, normalize=False, dropout_p=0.4),
            nn.Linear(512, 1)
        )

    def forward(self, imgs, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat(
            (imgs.view(imgs.size(0), -1), self.label_embedding(labels)), -1
        )
        validity = self.model(d_in)
        return validity
