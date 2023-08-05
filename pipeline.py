import os
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

# create INT8_Calib_Unlable_Dataset
# root = "/media/nvidia/T7 Shield/Orin_T906G/dataset/dataset/images/val"
# dirs = os.listdir(root)
# for image in tqdm(dirs):
#     path = os.path.join(root, image)
#     with open('calib_dataset.txt','a') as f:
#         f.write(f'{path}\n')


# create valid_label.text
# /home/rjg/dataset2/images/val/IMG_20221013_125157_aug3.jpg

# root1 = "/home/rjg/dataset2/images/val"
# root2 = "/home/rjg/dataset2/masks"

root1 = "/media/nvidia/T7 Shield/Orin_T906G/dataset/dataset/images/val"
root2 = "/media/nvidia/T7 Shield/Orin_T906G/dataset/dataset/masks"
dirs = os.listdir(root1)
for img_name in tqdm(dirs):
    if img_name.split('.')[0][-4:-1] == 'aug':
        mask_name = img_name.split('.')[0][:-5] + '.png'
    else:
        mask_name = img_name.split('.')[0] + '.png'
    with open('valid_label.txt','a') as f:
        f.write(f'{os.path.join(root2, mask_name)}\n')
