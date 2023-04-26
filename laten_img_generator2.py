import torch
  ## Imaging  library 
from PIL import Image 
from torchvision import transforms as tfms  
## Basic libraries 
import numpy as np 
#%matplotlib inline  
## Loading a VAE model 
from diffusers import AutoencoderKL 
from torchvision.utils import save_image
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

def load_image(p):
   '''     
   Function to load images from a defined path     
   '''    
   img = Image.open(p).convert('RGB').resize((256,256))
#    img = Image.open(p).convert('RGB').resize((64,64))
#    img = img.putalpha(0)
   return tfms.ToTensor()(img)

def encode_img(input_img):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def decode_img(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach()
    image = torch.squeeze(image)
    image = tfms.ToPILImage()(image).convert('RGBA')
    image = tfms.ToTensor()(image)
    return image


import os

# specify the img directory path
path = "/home/rbasiri/Dataset/GAN/test/DFU/"
savepath = "/home/rbasiri/Dataset/GAN/test/DFU/latenpic"

# list files in img directory
files = os.listdir(path)

for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = path + file

        # load file as image...
        img = load_image(img_path)
        latent = encode_img(img)
        latent_img = decode_img(latent)
        # latent_save_path = os.path.join(savepath, file[:-4]+'.npy')
        # np.save(latent_save_path, latent.cpu().detach().numpy())
        latent_save_path = os.path.join(savepath, file[:-4]+'.png')
        save_image(latent_img,latent_save_path)

# latent_save_path = os.path.join(test_opts.exp_dir, 'latent_code_%05d.npy'%global_i)
# latent_save_path = os.path.join(savepath, 'test.npy')
# np.save(latent_save_path, latent_img.cpu().numpy())