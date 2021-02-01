# AnimeGAN-torch

## This repo is to replicate results from AnimeGANv2 in Pytorch

This branch is to implement NOGAN training from [Deoldify](https://github.com/jantic/DeOldify#what-is-nogan). Since GAN training itself is unstable, it is desirable to have as minimal GAN training as possible yet still have the effects of GAN. NOGAN works well in image colourization, which is a task in image-to-image translation, intuitively it should also apply to other image-to-image translation tasks. Feature Loss of NOGAN training of AnimeGAN uses same loss in [AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/Chen2020_Chapter_AnimeGAN.pdf). The training of NOGAN happens in notebooks where the training is moderated heavily, with a lot of checkpointing and trial and error.

This branch adds FP16 training form [Nvidia Apex](https://github.com/NVIDIA/apex). Refer to the docs on different levels set to training. FP16 training is not applied to GAN training.
