import torch
from random import sample


class ImageBuffer():

    def __init__(self,
                 pool_size: int = 64,
                 batch_size: int = 16):

        self.pool_size = pool_size
        self.batch_size = batch_size
        self.num_images = 0
        self.images = []

    def __len__(self):
        return self.num_images

    def add_images(self, images: torch.Tensor):  # images.shape=[ , , , ]
        for i in range(images.shape[0]):
            assert self.num_images <= self.pool_size, "Buffer exceeded Buffer Size"
            self.images.append(images[i].cpu())
            self.num_images += 1

    def query(self, images: torch.Tensor):
        sample_index = sample(range(self.num_images), images.shape[0])
        sample_images = torch.stack([self.images[i] for i in sample_index])
        for i in range(len(sample_index)):
            self.images[sample_index[i]] = images[i].cpu()
        return sample_images
