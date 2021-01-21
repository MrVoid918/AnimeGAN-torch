import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn.functional as F

import kornia

#Many losses in this model require passing through VGG19, define once in init to pass on, rather than creating many classes
#to load more than once into memory

class VGGLosses(nn.Module):

  def __init__(self, resize = False, normalize_mean_std = True, device = 'cpu'):
    super(VGGLosses, self).__init__()

    self.model = models.vgg19(pretrained = True).features[:26].to(device).eval()  #features[:26] get to conv4_4
    for p in self.model.parameters():
      p.requires_grad = False

    self.blocks = nn.ModuleList([self.model[0],
                                 self.model[2],
                                 self.model[4:6]])

    self.inv_norm = transforms.Normalize([-1, -1, -1], [2., 2., 2.])   #[-1 , 1] -> [0, 1]
    self.inv_gray_transform = transforms.Compose([transforms.Normalize([-1, -1, -1], [2., 2., 2.]),
                                         transforms.Grayscale(num_output_channels = 3)])

    self.normalize_mean_std = normalize_mean_std
    if self.normalize_mean_std:
      self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1))
      self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1))

    self.resize = resize

  def gram_matrix(self, input):
      a, b, c, d = input.size()

      features = input.view(a * b, c * d)

      G = torch.mm(features, features.t())

      return G.div(a * b * c * d)

  def content_loss(self, input, target):

    '''real image & generated image'''

    if input.shape[1] != 3:
      input = input.repeat(1, 3, 1, 1)
      target = target.repeat(1, 3, 1, 1)
    '''
    if self.normalize_mean_std:
      input = (input-self.mean) / self.std
      target = (target-self.mean) / self.std
    '''
    input = self.model(input)
    target = self.model(target)

    return F.l1_loss(input, target)

  def style_loss(self, input, target, grayscale = False, luma = True):

    '''style image & generated image'''

    if luma:
      input = self.inv_norm(input)
      target = self.inv_norm(target)
      input = kornia.rgb_to_ycbcr(input)[:,0].unsqueeze(1)
      target = kornia.rgb_to_ycbcr(target)[:,0].unsqueeze(1)

    elif grayscale:
      input = self.inv_gray_transform(input)
      target = self.inv_gray_transform(input)
    '''
    elif self.normalize_mean_std:
      input = (input - self.mean) / self.std
      target = (target - self.mean) / self.std
    '''
    if input.shape[1] != 3:
      input = input.repeat(1, 3, 1, 1)
      target = target.repeat(1, 3, 1, 1)

    input = self.model(input)
    input_gram = self.gram_matrix(input)
    target = self.model(target)
    target_gram = self.gram_matrix(target)

    return F.l1_loss(input_gram, target_gram)

  def perceptual_loss(self, input, target, init_weight = 1 / 32.):

    r'''ESTHER
        real image & transformed image'''

    if input.shape[1] != 3:
      input = input.repeat(1, 3, 1, 1)
      target = target.repeat(1, 3, 1, 1)
    '''
    if self.normalize_mean_std:
      input = (input-self.mean) / self.std
      target = (target-self.mean) / self.std
    '''
    if self.resize:
      input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
      target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

    weight = init_weight

    total_loss = 0

    for block in self.blocks:
      input = block(input)
      target = block(target)
      loss = F.l1_loss(input, target) * weight
      weight *= 2.
      input = F.relu(input)   #we manually pass convolved image thorugh relu, instead of passing through blocks
      target = F.relu(target)  #this is because it is set to inplace, hence will cause error
      total_loss += loss

    return total_loss

  def reconstruction_loss(self, input, target):
    input = input * 0.5 + 0.5
    target = target * 0.5 + 0.5
    input = kornia.rgb_to_ycbcr(input)
    target = kornia.rgb_to_ycbcr(target)
    loss = F.l1_loss(input[:, 0], target[:, 0])
    loss += F.smooth_l1_loss(input[:, 1:], target[:, 1:])

    return loss

  def total_variation_loss(self, input):
    reg_loss = torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
    torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
    return reg_loss
