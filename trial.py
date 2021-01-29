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
from torch.optim.lr_scheduler import OneCycleLR
import json

import data_transform as tr
from dataset import Dataset
from weights_init import weights_init
from model.discriminator import PatchDiscriminator
from model.generator import Generator
from optimizers import GANOptimizer
from loss import Loss
from meter import AverageMeter, LossMeters


class Trial:

    def __init__(self,
                 data_dir: str = './dataset',
                 log_dir: str = './logs',
                 device: str = "cuda:0",
                 batch_size: int = 2,
                 init_lr: float = 0.5,
                 G_lr: float = 0.0004,
                 D_lr: float = 0.0008,
                 init_training_epoch: int = 10,
                 train_epoch: int = 10,
                 optim_type: str = "ADAM",
                 pin_memory: bool = True,
                 grad_set_to_none: bool = True):

        # self.config = config
        self.data_dir = data_dir

        self.dataset = Dataset(root=data_dir + "/Shinkai",
                               style_transform=tr.transform,
                               smooth_transform=tr.transform)

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=pin_memory)

        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        self.G = Generator().to(self.device)
        self.D = PatchDiscriminator().to(self.device)

        self.init_model_weights()

        self.optimizer_G = GANOptimizer(optim_type, self.G.parameters(),
                                        lr=G_lr, betas=(0.5, 0.999), amsgrad=True)
        self.optimizer_D = GANOptimizer(optim_type, self.D.parameters(),
                                        lr=D_lr, betas=(0.5, 0.999), amsgrad=True)

        self.loss = Loss(device=self.device).to(self.device)

        self.init_lr = init_lr
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.grad_set_to_none = grad_set_to_none

        self.writer = tensorboard.SummaryWriter(log_dir=log_dir)
        self.init_train_epoch = init_training_epoch
        self.train_epoch = train_epoch

        self.init_time = None

    def init_model_weights(self):
        self.G.apply(weights_init)
        self.D.apply(weights_init)

    def init_train(self, con_weight: float = 1.0):

        test_img = self.get_test_image()
        meter = AverageMeter("Loss")
        self.writer.flush()
        lr_scheduler = OneCycleLR(self.optimizer_G,
                                  max_lr=0.9999,
                                  steps_per_epoch=len(self.dataloader),
                                  epochs=self.init_train_epoch)

        for g in self.optimizer_G.param_groups:
            g['lr'] = self.init_lr

        for epoch in tqdm(range(self.init_train_epoch)):

            meter.reset()

            for i, (style, smooth, train) in enumerate(self.dataloader, 0):
                # train = transform(test_img).unsqueeze(0)
                self.G.zero_grad(set_to_none=self.grad_set_to_none)
                train = train.to(self.device)

                generator_output = self.G(train)
                # content_loss = loss.reconstruction_loss(generator_output, train) * con_weight
                content_loss = self.loss.content_loss(generator_output, train) * con_weight
                # content_loss = F.mse_loss(train, generator_output) * con_weight
                content_loss.backward()
                self.optimizer_G.step()
                lr_scheduler.step()

                meter.update(content_loss.detach())

            self.writer.add_scalar(f"Loss : {self.init_time}", meter.sum.item(), epoch)
            self.write_weights(epoch + 1, write_D=False)
            self.eval_image(epoch, f'{self.init_time} reconstructed img', test_img)

        for g in self.optimizer_G.param_groups:
            g['lr'] = self.G_lr

        # self.save_trial(self.init_train_epoch, "init")

    def eval_image(self, epoch: int, caption, img):
        """Feeds in one single image to process and save."""
        self.G.eval()
        styled_test_img = tr.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            styled_test_img = self.G(styled_test_img)
            styled_test_img = styled_test_img.to('cpu').squeeze()
        self.write_image(styled_test_img, caption, epoch + 1)
        self.writer.flush()
        self.G.train()

    def write_image(self,
                    image: torch.Tensor,
                    img_caption: str = "sample_image",
                    step: int = 0):

        inv_norm = transforms.Normalize([-1, -1, -1], [2., 2., 2.])
        image = inv_norm(image)  # [-1, 1] -> [0, 1]
        image *= 255.  # [0, 1] -> [0, 255]
        image = image.permute(1, 2, 0).to(dtype=torch.uint8)
        self.writer.add_image(img_caption, image, step, dataformats='HWC')
        self.writer.flush()

    def write_weights(self, epoch: int, write_D=True, write_G=True):

        if write_D:
            for name, weight in self.D.named_parameters():
                if 'depthwise' in name or 'pointwise' in name:
                    self.writer.add_histogram(
                        f"Discriminator {name} {self.init_time}", weight, epoch)
                    self.writer.add_histogram(
                        f"Discriminator {name}.grad {self.init_time}", weight.grad, epoch)
                    self.writer.flush()

        if write_G:
            for name, weight in self.G.named_parameters():
                self.writer.add_histogram(f"Generator {name} {self.init_time}", weight, epoch)
                self.writer.add_histogram(
                    f"Generator {name}.grad {self.init_time}", weight.grad, epoch)
                self.writer.flush()

    def train_1(self,
                adv_weight: float = 300.,
                con_weight: float = 1.5,
                gra_weight: float = 3.,
                col_weight: float = 10.,):

        test_img_dir = Path(self.data_dir).joinpath('test/test_photo256').resolve()
        test_img_dir = random.choice(list(test_img_dir.glob('**/*')))
        test_img = Image.open(test_img_dir)
        self.writer.add_image(f'test image {self.init_time}',
                              np.asarray(test_img), dataformats='HWC')
        self.writer.flush()

        for epoch in tqdm(range(self.train_epoch)):

            for i, (style, smooth, train) in enumerate(self.dataloader, 0):

                self.D.zero_grad()
                style = style.to(self.device)
                smooth = smooth.to(self.device)
                train = train.to(self.device)

                # style image to discriminator(Not Gram Matrix Loss)
                style_loss_value = self.D(style).view(-1)
                generator_output = self.G(train)
                # generated image to discriminator
                real_output = self.D(generator_output.detach()).view(-1)
                # greyscale_output = D(transforms.functional.rgb_to_grayscale(train, num_output_channels=3)).view(-1) #greyscale adversarial loss
                gray_train = tr.inv_gray_transform(train)
                greyscale_output = self.D(gray_train).view(-1)
                smoothed_loss = self.D(smooth).view(-1)  # smoothed image loss
                # loss_D_real = adversarial_loss(output, label)

                dis_adv_loss = adv_weight * (torch.pow(style_loss_value - 1, 2).mean() +
                                             torch.pow(real_output, 2).mean())
                dis_gray_loss = torch.pow(greyscale_output, 2).mean()
                dis_edge_loss = torch.pow(smoothed_loss, 2).mean()
                discriminator_loss = dis_adv_loss + dis_gray_loss + dis_edge_loss
                discriminator_loss.backward()
                self.optimizer_D.step()

                if i % 200 == 0 and i != 0:
                    self.writer.add_scalars(f'{self.init_time} Discriminator losses',
                                            {'adversarial loss': dis_adv_loss.item(),
                                             'grayscale loss': dis_gray_loss.item(),
                                             'edge loss': dis_edge_loss.item()},
                                            i + epoch * len(self.dataloader))
                    self.writer.flush()

                real_output = self.D(generator_output).view(-1)
                per_loss = self.loss.perceptual_loss(train, generator_output)  # loss for G
                style_loss = self.loss.style_loss(generator_output, style)
                content_loss = self.loss.content_loss(generator_output, train)
                recon_loss = self.loss.reconstruction_loss(generator_output, train)
                tv_loss = self.loss.total_variation_loss(generator_output)

                '''
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epoch, i, len(data_loader),
                      loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))'''

                self.G.zero_grad()
                gen_adv_loss = adv_weight * torch.pow(real_output - 1, 2).mean()
                gen_con_loss = con_weight * content_loss
                gen_sty_loss = gra_weight * style_loss
                gen_rec_loss = col_weight * recon_loss
                gen_per_loss = per_loss
                gen_tv_loss = tv_loss
                generator_loss = gen_adv_loss + gen_con_loss + gen_sty_loss + gen_rec_loss + gen_per_loss
                generator_loss.backward()
                self.optimizer_G.step()

                if i % 200 == 0 and i != 0:

                    self.writer.add_scalars(f'generator losses {self.init_time}',
                                            {'adversarial loss': gen_adv_loss.item(),
                                             'content loss': gen_con_loss.item(),
                                             'style loss': gen_sty_loss.item(),
                                             'reconstruction loss': gen_rec_loss.item(),
                                             'perceptual loss': gen_per_loss.item()},
                                            i + epoch * len(self.dataloader))
                    self.writer.flush()

            self.write_weights(epoch + 1)
            self.eval_image(epoch, f'{self.init_time} style img', test_img)

    def train_2(self,
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
                        self.loss.perceptual_loss(train, generator_output)
                    gen_adv_loss = adv_weight * torch.pow(real_output - 1, 2).mean()
                    gen_loss = gen_adv_loss + per_loss
                    gen_loss.backward()
                self.optimizer_G.step()

                if i % 200 == 0 and i != 0:
                    self.writer.add_scalars(f'generator losses  {self.init_time}',
                                            {'adversarial loss': dis_adv_loss.item(),
                                             'Generator adversarial loss': gen_adv_loss.item(),
                                             'perceptual loss': per_loss.item()}, i + epoch * len(self.dataloader))
                    self.writer.flush()

            if total_dis_loss > threshold and not keep_constant:
                perception_weight += 0.05
            else:
                keep_constant = True

            self.writer.add_scalar(
                f'total discriminator loss {self.init_time}', total_dis_loss, i + epoch * len(self.dataloader))

            self.write_weights()
            self.G.eval()

            styled_test_img = tr.transform(test_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                styled_test_img = self.G(styled_test_img)

            styled_test_img = styled_test_img.to('cpu').squeeze()
            self.write_image(styled_test_img, f'styled image {self.init_time}', epoch + 1)

            self.G.train()

    def __call__(self):
        self.init_train()
        self.train_1()

    def save_trial(self, epoch: int, train_type: str):
        save_dir = Path(f"Trial_{train_type}_{self.init_time}.pt")
        training_details = {"epoch": epoch,
                            "generator": {"generator_state_dict": self.G.state_dict(),
                                          "optimizer_G_state_dict": self.optimizer_G.state_dict(), },
                            "discriminator": {"discriminator_state_dict": self.D.state_dict(),
                                              "optimizer_D_state_dict": self.optimizer_D.state_dict()}}

        torch.save(training_details, save_dir.as_posix())

    def Generator_NOGAN(self,
                        epoch: int = 1,
                        style_weight: float = 20.,
                        content_weight: float = 1.2,
                        recon_weight: float = 10.,
                        tv_weight: float = 1.):
        """Training Generator in NOGAN manner (Feature Loss only)."""
        test_img = self.get_test_image()
        self.writer.flush()
        lr_scheduler = OneCycleLR(self.optimizer_G,
                                  max_lr=0.5,
                                  steps_per_epoch=len(self.dataloader),
                                  epochs=epoch)
        meter = LossMeters('style_loss',
                           'content_loss')
        for epoch in tqdm(range(epoch)):

            meter.reset()

            for i, (style, smooth, train) in enumerate(self.dataloader, 0):
                # train = transform(test_img).unsqueeze(0)
                self.G.zero_grad(set_to_none=self.grad_set_to_none)
                train = train.to(self.device)
                style = style.to(self.device)

                generator_output = self.G(train)
                #style_loss = self.loss.style_loss(generator_output, style) * style_weight
                content_loss = self.loss.content_loss(generator_output, train) * content_weight
                #recon_loss = self.loss.reconstruction_loss(generator_output, train) * recon_weight
                #tv_loss = self.loss.total_variation_loss(generator_output) * tv_weight
                total_loss = content_loss
                total_loss.backward()
                self.optimizer_G.step()
                lr_scheduler.step()

                meter.update(content_loss.detach())

            self.writer.add_scalars(f'{self.init_time} NOGAN generator losses',
                                    meter.as_dict('sum'),
                                    epoch)
            self.write_weights(epoch + 1, write_D=False)
            self.eval_image(epoch, f'{self.init_time} reconstructed img', test_img)

    def Discriminator_NOGAN(self, epoch: int, adv_weight: float = 1.0):
        lr_scheduler = OneCycleLR(self.optimizer_D,
                                  max_lr=1e-2,
                                  steps_per_epoch=len(self.dataloader),
                                  epochs=epoch)
        meter = LossMeters('real_adv_loss', 'fake_adv_loss')

        for epoch in tqdm(range(epoch)):

            meter.reset()

            for i, (style, smooth, train) in enumerate(self.dataloader, 0):
                # train = transform(test_img).unsqueeze(0)
                self.D.zero_grad(set_to_none=self.grad_set_to_none)
                train = train.to(self.device)
                style = style.to(self.device)

                generator_output = self.G(train)
                real_adv_loss = self.D(style).view(-1)
                fake_adv_loss = self.D(generator_output.detach()).view(-1)
                real_adv_loss = torch.pow(real_adv_loss - 1, 2).mean() * adv_weight
                fake_adv_loss = torch.pow(fake_adv_loss, 2).mean() * adv_weight
                total_loss = real_adv_loss + fake_adv_loss
                total_loss.backward()
                self.optimizer_D.step()
                lr_scheduler.step()

                meter.update(real_adv_loss.detach(), fake_adv_loss.detach())

            self.writer.add_scalars(f'{self.init_time} NOGAN discriminator loss',
                                    meter.as_dict('sum'),
                                    epoch)
            self.writer.flush()

    def get_test_image(self):
        """Get random test image."""
        test_img_dir = Path(self.data_dir).joinpath('test/test_photo256').resolve()
        test_img_dir = random.choice(list(test_img_dir.glob('**/*')))
        test_img = Image.open(test_img_dir)
        self.init_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.writer.add_image(f'{self.init_time} sample_image',
                              np.asarray(test_img), dataformats='HWC')

        return test_img
