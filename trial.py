import random
from pathlib import Path
import datetime
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from torchvision import transforms
import json

import data_transform as tr
from dataset import Dataset
from weights_init import weights_init
from model.discriminator import PatchDiscriminator
from model.generator import Generator
from optimizers import GANOptimizer
from loss import VGGLosses


class Trial:

    def __init__(self,
                 data_dir: str = './dataset',
                 device="cuda:0",
                 batch_size = 2,
                 init_lr: float = 0.05,
                 G_lr: float = 0.0004,
                 D_lr: float = 0.0004,
                 init_training_epoch: int = 10,
                 train_epoch: int = 10,):

        # self.config = config
        self.data_dir = data_dir

        self.dataset = Dataset(root='./dataset/Shinkai',
                               style_transform=tr.transform,
                               smooth_transform=tr.transform)

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        self.G = Generator().to(self.device)
        self.D = PatchDiscriminator().to(self.device)

        self.init_model_weights()

        self.optimizer_G = GANOptimizer("ADAM", self.G.parameters(), lr=G_lr, betas=(0.5, 0.999), amsgrad = True)
        self.optimizer_D = GANOptimizer("ADAM", self.D.parameters(), lr=D_lr, betas=(0.5, 0.999), amsgrad = True)

        self.vggloss = VGGLosses(device=self.device).to(self.device)

        self.init_lr = init_lr
        self.G_lr = G_lr
        self.D_lr = D_lr

        self.writer = tensorboard.SummaryWriter(log_dir='./logs')
        self.init_train_epoch = init_training_epoch
        self.train_epoch = train_epoch

        self.init_time = None


    def init_model_weights(self):
        self.G.apply(weights_init)
        self.D.apply(weights_init)

    def init_train(self, con_weight : float = 1.0):

        test_img_dir = Path(self.data_dir).joinpath('./test/test_photo256').resolve()
        test_img_dir = random.choice(list(test_img_dir.glob('**/*')))
        test_img = Image.open(test_img_dir)
        self.init_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.writer.add_image(f'sample_image {self.init_time}',
                              np.asarray(test_img), dataformats='HWC')
        self.writer.flush()

        for g in self.optimizer_G.param_groups:
            g['lr'] = self.init_lr

        for epoch in tqdm(range(self.init_train_epoch)):

            total_loss = 0

            for i, (style, smooth, train) in enumerate(self.dataloader, 0):
                # train = transform(test_img).unsqueeze(0)
                self.G.zero_grad()
                train = train.to(self.device)

                generator_output = self.G(train)
                # content_loss = vggloss.reconstruction_loss(generator_output, train) * con_weight
                content_loss = self.vggloss.content_loss(generator_output, train) * con_weight
                # content_loss = F.mse_loss(train, generator_output) * con_weight

                content_loss.backward()
                self.optimizer_G.step()

                total_loss += content_loss

                print(i)

            self.writer.add_scalar(f"Loss : {self.init_time}", total_loss.item(), epoch)

            for name, weight in self.G.named_parameters():
                self.writer.add_histogram(f"{name} {self.init_time}", weight, epoch)
                self.writer.add_histogram(f"{name}.grad {self.init_time}", weight.grad, epoch)
                self.writer.flush()

            self.eval_image(epoch, self.init_time, test_img)

        for g in optimizer_G.param_groups:
            g['lr'] = 0.0002

        self.save_trial(self.init_training_epoch, "init")

    def eval_image(self, epoch: int, current_time, img):
        self.G.eval()

        styled_test_img = tr.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            styled_test_img = self.G(styled_test_img)
            styled_test_img = styled_test_img.to('cpu').squeeze()
        write_image(self.writer, styled_test_img, f'reconstructed img {current_time}', epoch + 1)
        self.writer.flush()
        self.G.train()

    def write_image(self,
                    image : torch.Tensor,
                    img_caption : str = "sample_image",
                    step : int = 0):

        inv_norm = transforms.Normalize([-1, -1, -1], [2., 2., 2.])
        image = inv_norm(image)   #[-1, 1] -> [0, 1]
        image *= 255.             #[0, 1] -> [0, 255]
        image = image.permute(1, 2, 0).to(dtype = torch.uint8)
        self.writer.add_image(img_caption, image, step, dataformats= 'HWC')
        self.writer.flush()

  #assert torch.min(image).item() >= -1. and torch.max(image).item() <= 1.
  inv_norm = transforms.Normalize([-1, -1, -1], [2., 2., 2.])
  image = inv_norm(image)   #[-1, 1] -> [0, 1]
  image *= 255.             #[0, 1] -> [0, 255]
  image = image.permute(1, 2, 0).to(dtype = torch.uint8)
  writer.add_image(img_caption, image, step, dataformats= 'HWC')
  writer.flush()

    def train(self,
              adv_weight: float = 1.0,
              threshold: float = 3.,
              G_train_iter: int = 1,
              D_train_iter: int = 1):  # if threshold is 0., set to half of adversarial loss

        test_img_dir = Path(self.data_dir).joinpath('test', 'test_photo256').resolve()
        test_img_dir = random.choice(list(test_img_dir.glob('**/*')))
        test_img = Image.open(test_img_dir)

        if self.init_time is None:
            self.init_time = datetime.datetime.now().strftime("%H:%M:%S")

        self.writer.add_image(f'sample_image {self.init_time}',
                              np.asarray(test_img), dataformats='HWC')
        self.writer.flush()

        perception_weight = 0.
        keep_constant = False

        for epoch in tqdm(range(self.train_epoch)):

            total_dis_loss = 0.

            for i, (style, smooth, train) in enumerate(self.dataloader, 0):

                self.D.zero_grad()

                train = train.to(self.device)
                style = style.to(self.device)
                # smooth = smooth.to(device)

                for _ in range(D_train_iter):
                    style_loss_value = self.D(style).view(-1)
                    generator_output = self.G(train)
                    real_output = self.D(generator_output.detach()).view(-1)
                    dis_adv_loss = adv_weight * \
                        (torch.pow(style_loss_value - 1, 2).mean() + torch.pow(real_output, 2).mean())
                    total_dis_loss += dis_adv_loss.item()
                    dis_adv_loss.backward()
                self.optimizer_D.step()

                self.G.zero_grad()
                for _ in range(G_train_iter):
                    generator_output = self.G(train)
                    real_output = self.D(generator_output).view(-1)
                    per_loss = perception_weight * \
                        self.vggloss.perceptual_loss(train, generator_output)
                    gen_adv_loss = adv_weight * torch.pow(real_output - 1, 2).mean()
                    gen_loss = gen_adv_loss + per_loss
                    gen_loss.backward()
                self.optimizer_G.step()

                print(i)

                if i % 200 == 0 and i != 0:
                    self.writer.add_scalars(f'generator losses  {self.init_time}',
                                            {'adversarial loss': dis_adv_loss.item(),
                                             'Generator adversarial loss': gen_adv_loss.item(),
                                             'perceptual loss': per_loss.item()}, i + epoch * len(self.dataloader))
                    self.writer.flush()

            if total_dis_loss > threshold and not keep_constant:
                perception_weight += 0.05
            else:
                keep_constant=True

            self.writer.add_scalar(
                f'total discriminator loss {self.init_time}', total_dis_loss, i + epoch * len(self.dataloader))

            for name, weight in D.named_parameters():
                if 'depthwise' in name or 'pointwise' in name:
                    self.writer.add_histogram(
                        f"Discriminator {name} {self.init_time}", weight, epoch)
                    self.writer.add_histogram(
                        f"Discriminator {name}.grad {self.init_time}", weight.grad, epoch)
                    self.writer.flush()

            for name, weight in G.named_parameters():
                self.writer.add_histogram(f"Generator {name} {self.init_time}", weight, epoch)
                self.writer.add_histogram(
                    f"Generator {name}.grad {self.init_time}", weight.grad, epoch)
                self.writer.flush()

            self.G.eval()

            styled_test_img = transform(test_img).unsqueeze(0).to(device)
            with torch.no_grad():
                styled_test_img = self.G(styled_test_img)

            styled_test_img = styled_test_img.to('cpu').squeeze()
            write_image(self.writer, styled_test_img, f'styled image {self.init_time}', epoch + 1)

            self.G.train()

    def __call__(self):
        self.init_train()
        self.train()

    def save_trial(self, epoch: int, train_type: str):
        save_dir = Path(f"./Trial_{train_type}_{self.init_time}.pt")
        training_details = {"epoch": epoch,
                            "generator": {"generator_state_dict": self.G.state_dict(),
                                          "optimizer_G_state_dict": self.optimizer_G.state_dict(), },
                            "discriminator": {"discriminator_state_dict": self.D.state_dict(),
                                              "optimizer_D_state_dict": self.optimizer_D.state_dict()}}

        torch.save(training_details, save_dir)
        training_details["saved_dir"] = save_dir.as_posix()

        with open(self.config, "r+") as file:
            data = json.load(file)
            data.update(training_details)
            file.seek(0)
            json.dump(data, file)
