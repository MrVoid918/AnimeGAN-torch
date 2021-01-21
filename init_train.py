import random
from pathlib import Path
import datetime
from PIL import Image
import tqdm
import numpy as np

import torch.nn as nn
import torch

def init_train(init_train_epoch : int = 1,
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
