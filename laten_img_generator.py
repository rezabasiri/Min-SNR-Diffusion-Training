import torch
  ## Imaging  library 
from PIL import Image 
from torchvision import transforms as tfms  
## Basic libraries 
import numpy as np 
#%matplotlib inline  
## Loading a VAE model 
from diffusers import AutoencoderKL 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cuda")

def load_image(p):
   '''     
   Function to load images from a defined path     
   '''    
   return Image.open(p).convert('RGB').resize((256,256))
def pil_to_latents(image):
    '''     
    Function to convert image to latents     
    '''     
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215     
    return init_latent_dist  
def latents_to_pil(latents):     
    '''     
    Function to convert latents to images     
    '''     
    latents = (1 / 0.18215) * latents     
    with torch.no_grad():         
        image = vae.decode(latents).sample     
    
    image = (image / 2 + 0.5).clamp(0, 1)     
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
    images = (image * 255).round().astype("uint8")     
    pil_images = [Image.fromarray(image) for image in images]        
    return pil_images

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
        latent = pil_to_latents(img)
        latent_img = latents_to_pil(latent)
        # latent_save_path = os.path.join(savepath, file[:-4]+'.npy')
        # np.save(latent_save_path, latent.cpu().detach().numpy())
        latent_save_path = os.path.join(savepath, file[:-4]+'.jpg')
        latent_img.save(latent_save_path)

# latent_save_path = os.path.join(test_opts.exp_dir, 'latent_code_%05d.npy'%global_i)
# latent_save_path = os.path.join(savepath, 'test.npy')
# np.save(latent_save_path, latent_img.cpu().numpy())