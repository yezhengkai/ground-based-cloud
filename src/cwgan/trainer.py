# Training conditional Wasserstein Generative Adversarial Network with Gradient Penalty
# References:
#   https://github.com/u7javed/Conditional-WGAN-GP/blob/master/train.py
#   https://github.com/gcucurull/cond-wgan-gp/blob/master/main.py
#   https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import save_image, make_grid

from .models import Discriminator, Generator


class Trainer:
    def __init__(self, img_shape, num_classes, embedding_dim,
                 latent_size=100, device=None):
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # define models
        self.latent_size = 100

        self.discriminator = Discriminator(
            img_shape, num_classes, embedding_dim
        ).to(device)
        self.generator = Generator(
            img_shape, latent_size, num_classes, embedding_dim
        ).to(device)

    def compute_gradient_penalty(self, real_samples, fake_samples, labels):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(
            np.random.random((real_samples.size(0), 1, 1, 1))
        ).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, labels)
        fake = torch.Tensor(
            real_samples.shape[0], 1
        ).fill_(1.0).to(self.device)
        fake.requires_grad = False  # default is False
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(gradients[0].size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def show_sample_image(self):
        nrow = self.num_classes
        with torch.no_grad():
            # Sample noise
            z = torch.randn(nrow ** 2, self.latent_size).to(self.device)
            # Get labels ranging from 0 to n_classes for n rows
            labels = torch.LongTensor(
                torch.arange(nrow).repeat(nrow)
            ).to(self.device)
            gen_imgs = self.generator(z, labels)
        image_grid = make_grid(gen_imgs.cpu().detach(),
                               nrow=nrow, normalize=True)
        _, ax = plt.subplots(figsize=(12, 12))
        plt.axis('off')
        ax.imshow(image_grid .permute(1, 2, 0))

    def save_sample_image(self, nrow, filename):
        with torch.no_grad():
            # Sample noise
            z = torch.randn(nrow ** 2, self.latent_size).to(self.device)
            # Get labels ranging from 0 to n_classes for n rows
            labels = torch.LongTensor(
                torch.arange(nrow).repeat(nrow)
            ).to(self.device)
            gen_imgs = self.generator(z, labels)
        save_image(gen_imgs, filename, nrow=nrow, normalize=True)

    def train(self, data_loader, epochs, saved_image_directory, saved_model_directory,
              save_model_every_epoch=3, train_gen_every_batch=5, lambda_gp=10,
              lr=0.0002, optimizer=None,):

        # set optimizer
        if optimizer is None:
            # default optimizer
            self.optimizer_d = optim.RMSprop(
                self.discriminator.parameters(), lr=lr)
            self.optimizer_g = optim.RMSprop(
                self.generator.parameters(), lr=lr)
        else:
            self.optimizer_d = optimizer
            self.optimizer_g = optimizer

        # create directory
        saved_image_directory = Path(saved_image_directory)
        saved_model_directory = Path(saved_model_directory)
        if not saved_image_directory.is_dir():
            saved_image_directory.mkdir(parents=True)
        if not saved_model_directory.is_dir():
            saved_model_directory.mkdir(parents=True)

        gen_loss_list = []
        dis_loss_list = []
        was_loss_list = []
        save_model_every_epoch = 1 if save_model_every_epoch < 1 else save_model_every_epoch
        train_gen_every_batch = 1 if train_gen_every_batch < 1 else train_gen_every_batch

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            running_gen_loss = 0
            _gen_dataset_size = 0
            running_dis_loss = 0
            cur_time = time.time()
            for i, (real_imgs, labels) in enumerate(data_loader):
                batch_size = len(real_imgs)

                real_imgs = real_imgs.type(torch.Tensor).to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)

                # Train Discriminator with Wasserstein Loss
                self.optimizer_d.zero_grad()

                # noises and labels as generator input
                z = torch.randn(
                    batch_size, self.latent_size).to(self.device)
                # generate a batch of fake images
                fake_imgs = self.generator(z, labels)

                # fake loss
                fake_pred = self.discriminator(fake_imgs, labels)
                d_loss_fake = torch.mean(fake_pred)

                # real loss
                real_pred = self.discriminator(real_imgs, labels)
                d_loss_real = -torch.mean(real_pred)  # negative

                # compute gradient penalty
                gp = self.compute_gradient_penalty(
                    real_imgs, fake_imgs, labels
                )

                # d_loss = d_loss_fake - d_loss_real
                d_loss = (d_loss_fake - d_loss_real) / 2
                was_loss = (d_loss_fake + d_loss_real) + lambda_gp * gp
                was_loss.backward()
                self.optimizer_d.step()

                # dis_loss += d_loss.item() / batch_size
                running_dis_loss += d_loss.item() * batch_size

                # Train the generator every `train_gen_every_batch` steps
                if i % train_gen_every_batch == 0:
                    self.optimizer_g.zero_grad()

                    z = torch.randn(
                        batch_size, self.latent_size).to(self.device)
                    # generate a batch of fake images
                    fake_imgs = self.generator(z, labels)
                    fake_pred = self.discriminator(fake_imgs, labels)
                    g_loss = -torch.mean(fake_pred)
                    g_loss.backward()
                    self.optimizer_g.step()

                    # gen_loss += g_loss.item() / batch_size
                    running_gen_loss += g_loss.item() * batch_size
                    _gen_dataset_size += batch_size

            cur_time = time.time() - cur_time

            epoch_gen_loss = running_gen_loss / _gen_dataset_size
            epoch_dis_loss = running_dis_loss / len(data_loader.dataset)
            print(
                "Epoch {}/{},    Gen Loss: {:.4f},   Dis Loss: {:.4f},   Was Loss: {:.4f}".format(
                    epoch, epochs, epoch_gen_loss, epoch_dis_loss, was_loss
                )
            )
            print(
                "Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining".format(
                    cur_time, (epochs - epoch) * (cur_time) / 3600
                )
            )
            gen_loss_list.append(epoch_gen_loss)
            dis_loss_list.append(epoch_dis_loss)
            was_loss_list.append(was_loss)

            # save models to model_directory
            if epoch % save_model_every_epoch == 0 or epoch == epochs:
                torch.save(
                    self.generator.state_dict(),
                    saved_model_directory / "generator_{}.pt".format(epoch),
                )
                torch.save(
                    self.discriminator.state_dict(),
                    saved_model_directory /
                    "discriminator_{}.pt".format(epoch),
                )

            # save sample images
            self.save_sample_image(self.num_classes,
                                   saved_image_directory
                                   / "epoch_{}_checkpoint.jpg".format(epoch))

        finish_time = time.time() - start_time
        print(
            "Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.".format(
                finish_time, finish_time / 3600
            )
        )

        return {
            "gen_loss": gen_loss_list,
            "dis_loss": dis_loss_list,
            "was_loss": was_loss_list
        }
