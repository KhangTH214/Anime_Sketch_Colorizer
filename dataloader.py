import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import random

def extract_dominant_colors(img, nclusters=9):
    """
    Apply K-means clustering to extract dominant colors from the input image.

    Args:
        img: Numpy array which has shape of (H, W, C).
        nclusters: Number of clusters (default = 9).

    Returns:
        color_palette: A list of 3D numpy arrays, where each array has the same shape as the input image.
        e.g. If input image has shape of (256, 256, 3) and nclusters is 4, the return color_palette is [color1, color2, color3, color4]
             and each component is (256, 256, 3) numpy array.
    """
    img_size = img.shape

    # Resize the image to speed up calculation
    small_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    sample = small_img.reshape((-1, 3))
    sample = np.float32(sample)

    # K-means clustering 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, _, centers = cv2.kmeans(sample, nclusters, None, criteria, 10, flags)
    centers = np.uint8(centers)
    
    color_palette = []
    for i in range(nclusters):
        dominant_color = np.zeros(img_size, dtype='uint8')
        dominant_color[:,:,:] = centers[i]
        color_palette.append(dominant_color)
    
    return color_palette

class PairImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, nclusters=9):
        super().__init__(root, transform)
        self.nclusters = nclusters

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        color_img = img.crop((0, 0, 512, img.height))
        edge_img = img.crop((512, 0, img.width, img.height))  
 
        seed = random.randint(0, 99999)
        torch.manual_seed(seed)
        color_img = self.transform(color_img)
        torch.manual_seed(seed)
        edge_img = self.transform(edge_img)

        color_palette = extract_dominant_colors(np.array(color_img), nclusters=self.nclusters)

        color_img = self.convert_to_tensor(color_img)
        edge_img = self.convert_to_tensor(edge_img)
        for i in range(len(color_palette)):
            color_palette[i] = self.convert_to_tensor(Image.fromarray(color_palette[i]))

        return edge_img, color_img, color_palette

    def convert_to_tensor(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
        ])
        img = transform(img)
        return img