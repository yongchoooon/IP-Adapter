import os
import torch
import random

from PIL import Image
from tqdm import tqdm

######################## Configuration ########################

from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterXL

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

sample_num = 5000
generated_image_path = 'generated/sdxl'
pascal_images_root_path = 'VOCdevkit/VOC2012/JPEGImages'

num_samples_per_seed = 1

###############################################################

images_list = sorted([pascal_images_root_path + '/' + path for path in os.listdir(pascal_images_root_path)])
len_images = len(images_list)

sampled_images_list = random.sample(images_list, sample_num)

os.makedirs(generated_image_path, exist_ok = True)

print('=======================================================')
print(f'PASCAL VOC 2012의 전체 이미지 갯수 : {len_images}')
print(f'PASCAL VOC 2012의 {sample_num}개의 이미지에 대한 생성을 시작합니다.')
print('=======================================================')

for image_path in tqdm(sampled_images_list):
    image = Image.open(image_path)
    image_name = image_path.split('/')[-1][:-4] # ex) 2008_002502

    images_42 = ip_model.generate(pil_image=image, num_samples=num_samples_per_seed, num_inference_steps=30, seed=42)
    images_421 = ip_model.generate(pil_image=image, num_samples=num_samples_per_seed, num_inference_steps=30, seed=421)

    for image_42 in images_42:
        image_42.save(f'{generated_image_path}/{image_name}_42.jpeg')

    for image_421 in images_421:
        image_421.save(f'{generated_image_path}/{image_name}_421.jpeg')

    