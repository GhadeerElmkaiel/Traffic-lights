import os
import sys
from PIL import Image
import numpy as np

TL_Label =  (250,170,30, 255)

current_path = os.path.abspath(os.getcwd())

depth_path = os.path.join(current_path, "data/Depth")
depth_deb_path = os.path.join(current_path, "data/DepthDebug")
gt_deb_path = os.path.join(current_path, "data/GTDebug")
rgb_path = os.path.join(current_path, "data/RGB")
res_path = os.path.join(current_path, "data/results")

image_name = "0000013.png"

mask = Image.open(os.path.join(gt_deb_path, image_name)) 
pixels = np.array(mask)
pixels_mask = np.array([int(pixels[i][j][0]==TL_Label[0] and pixels[i][j][1]==TL_Label[1]) for i in range(len(pixels)) for j in range(len(pixels[0]))])

pixels_mask = np.reshape(pixels_mask,(1080,1920))
pixels_mask = [pixels_mask, pixels_mask, pixels_mask, pixels_mask]
pixels_mask = np.swapaxes(pixels_mask, 0,2)
pixels_mask = np.swapaxes(pixels_mask, 1,0)
inv_mask = 1-pixels_mask

depth = Image.open(os.path.join(depth_path, image_name)) 
depth_pixels = np.array(depth)

rgb = Image.open(os.path.join(rgb_path, image_name)) 
rgb_pixels = np.array(rgb)

new_rgb_pixels = rgb_pixels*rgb_pixels + pixels_mask*TL_Label
new_rgb = Image.fromarray(new_rgb_pixels.astype(np.uint8)).save(current_path+"/test_rgb.png")
image = Image.open(current_path+"/data/results/0000217.png")
arr = np.array(image)
vals = np.unique(arr)
print(vals)