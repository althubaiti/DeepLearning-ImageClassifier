from torchvision import datasets, transforms
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
import seaborn as sn
import time
import os
import numpy as np
means = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
    
import torch

def cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def cat_to_list():
    category = cat_to_name()
    flower_list =[] 
    for key in range (len (category) ) :
        dic_key = str(key + 1)
        flower_list.append(category[dic_key])
    return flower_list


def classify_flower (preds):
    pclasses = []
    flower_list = cat_to_list()
    for pred in preds:
        pclasses.append( flower_list[pred - 1] )
    return pclasses 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    image  = image.resize((256,256))
    # 1- Crop the Image by 224 px :
    left_margin = (image.width - 224) / 2
    bottom_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    # 2- Convert Values : 
    image = (image - means) / std
    
     # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    #[width, height, channels]
    image = image.transpose(2, 0, 1)
    
    return image

def imshow(image, ax=None, title=None , normalize=True):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))
    #image = image * 255
    
    #Undo Conversion
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
 
    
    if title:
        plt.title(title)
    
    image = np.clip(image, 0, 1)
        
        
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    
    return ax

def predict(image_path, model, topk=5):
    
    
    # TODO: Implement the code to predict the class from an image file
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.cpu()
    image = process_image(image_path) 
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image = image_tensor.unsqueeze(0)
 
    
    softMax = model.forward(image)
    probs = torch.exp(softMax)
    
    top_probs, top_labs = probs.topk(5)
         
    
    # Convert from Tesors to Numpy arrays
    top_probs  = torch.topk(probs, topk)[0].tolist()[0] # probabilities
    idx_to_class = torch.topk(probs, topk)[1].tolist()[0] # index
    model.to(device);
        
    return top_probs, idx_to_class
