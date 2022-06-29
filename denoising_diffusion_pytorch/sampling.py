import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision.utils import save_image

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8)
).cuda

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda

diffusion = diffusion.load_state_dict('results/model-3.pt')

training_images = torch.randn(8, 3, 32, 32) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)

img = sampled_images[0]
save_image(img, 'img_sample.png')