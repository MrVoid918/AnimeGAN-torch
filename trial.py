import random
from pathlib import Path
import datetime
from PIL import Image
import tqdm
import numpy as np

import torch.nn as nn
import torch

import data_transform as tr
from dataset import Dataset
from weights_init import weights_init

class Trial:

    def __init__(self,
                 data_dir : str = './dataset',
                 device = "cuda:0",
                 init_lr : float = 0.05,
                 G_lr : float = 0.0004,
                 D_lr : float = 0.0004,
                 ):

        self.dataset = Dataset(root = 'dataset/Shinkai',
                               style_transform = transform,
                               smooth_transform = transform)

        self.dataloader = DataLoader(self.dataset,
                                     batch_size = 16,
                                     shuffle = True)

        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        self.G = Generator().to(self.device)
        self.D = PatchDiscriminator().to(self.device)

        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.optimizer_G = optim.Adam(self.G.parameters(), lr = 0.0001)   #Based on paper
        self.optimizer_D = optim.Adam(self.D.parameters(), lr = 0.0004)   #Based on paper

        self.init_lr = init_lr
        self.G_lr = G_lr
        self.D_lr = D_lr

        self.writer = tensorboard.SummaryWriter(log_dir = './logs')
        self.init_training_epoch = init_training_epoch


    def init_train(self, init_train_epoch : int = 1,
                    lr : float = 0.05,
                       con_weight : float = 1.5,
                       ):

          test_img_dir = Path('./dataset/test/test_photo256').resolve()
          test_img_dir = random.choice(list(test_img_dir.glob('**/*')))
          test_img = Image.open(test_img_dir)
          current_time = datetime.datetime.now().strftime("%H:%M:%S")
          writer.add_image('sample_image {}'.format(current_time), np.asarray(test_img), dataformats='HWC')
          writer.flush()

          for g in optimizer_G.param_groups:
              g['lr'] = lr

          for epoch in tqdm(range(init_train_epoch)):

            total_loss = 0

            for i, (style, smooth, train) in enumerate(dataloader, 0):
              #train = transform(test_img).unsqueeze(0)
              G.zero_grad()
              train = train.to(device)

              generator_output = G(train)
              #content_loss = vggloss.reconstruction_loss(generator_output, train) * con_weight
              content_loss = vggloss.content_loss(generator_output, train) * con_weight
              #content_loss = F.mse_loss(train, generator_output) * con_weight

              content_loss.backward()
              optimizer_G.step()

              total_loss += content_loss
              '''
                if not i % 200 and i != 0:
                  print("Loss: {}".format(content_loss.item()))
                  writer.add_scalar("trial_2_con_Loss", content_loss.item(), i + epoch * len(dataloader))'''

            writer.add_scalar("Loss : {}".format(current_time) , total_loss.item(), epoch)

            for name, weight in G.named_parameters():
                writer.add_histogram(f"{name} {current_time}", weight, epoch)
                writer.add_histogram(f"{name}.grad {current_time}", weight.grad, epoch)
                writer.flush()

            G.eval()

            styled_test_img = transform(test_img).unsqueeze(0).to(device)
            with torch.no_grad():
              styled_test_img = G(styled_test_img)
              styled_test_img = styled_test_img.to('cpu').squeeze()
            write_image(writer, styled_test_img, 'reconstructed img {}'.format(current_time), epoch + 1)
            writer.flush()
            G.train()

          for g in optimizer_G.param_groups:
            g['lr'] = 0.0002
