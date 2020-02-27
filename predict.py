# Image Classifier Part-2: Prediction
#PROGRAMMER: Thomas Innerebner
#Date created:29.01.2020
#Date revised:26.02.2020

import numpy as np
import matplotlib.pyplot as plt
import torch
from numba import cuda
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import time

import argparse

#TO DO: Parser arguments

parser = argparse.ArgumentParser()

parser.add_argument("--image_input", default="/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg", help="path to training data")
parser.add_argument("--gpu", default="gpu", help="trains on GPU", action="store")
parser.add_argument("--checkpoint", default= "/home/workspace/ImageClassifier/checkpoint.pth", help="Path to trained model")
parser.add_argument("--top_k", default=3, type=int, help="top categories")
parser.add_argument("--category_names", default= "/home/workspace/ImageClassifier/cat_to_name.json")
args = parser.parse_args()

# DONE: GPU 

if args.gpu == "gpu":
    power = "cuda"
else:
    power = "cpu"

# DONE: Parser

image_input = args.image_input
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names


#Done: Print settings:

print(power)                    
print(image_input)
print(checkpoint)
print(top_k)
print(category_names)

# DONE: Define modell structure

structures = {"vgg16":25088,
              "densenet121" : 1024,
              "alexnet" : 9216 }                    
                    
# DONE: Eingangsgrößen und Modell Auswahl (VGG13 Changes)
                    
def nn_setup(structure='vgg16',dropout = 0.4, hidden_units = 4096, learning_rate = 0.001):
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} ist kein valides Modell".format(structure))
        
    # Training aussetzen 
        
    for param in model.parameters():
        param.requires_grad = False

    # Classifier definieren
    
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_units)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_units, 1000)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(1000,103)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
       
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate )
        
        return model , optimizer ,criterion 

# Load model (Dropout rate changes)
def load_model(path = "/home/workspace/ImageClassifier/checkpoint.pth"):
    checkpoint = torch.load("/home/workspace/ImageClassifier/checkpoint.pth")
    structure = checkpoint['structure']
    hidden_units = checkpoint['hidden_units']
    model,_,_= nn_setup(structure, 0.4, hidden_units)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

load_model("/home/workspace/ImageClassifier/checkpoint.pth")

print("model succesfully loaded")

#DONE: Load cat names

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

#DONE : Process image:

def process_image(image):
    
    pil_image = Image.open(image)
    im_resized = pil_image.resize((224,224))
    
    np_image = np.array(im_resized)
    np_image = np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose(2, 0, 1)
    return np_image

#process_image(image_input)

#TO DO: Predict image:

def predict(image_input, model, top_k=3, power="cuda"):   
    model.to(power)
    img_pip = process_image(image_input)
    img_tensor = torch.from_numpy(img_pip)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()
    with torch.no_grad():
        output = model.forward(img_tensor.to(power))
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(top_k)

# TO DO: Load Model checkpoint

model = load_model(checkpoint)

image_filename = image_input.split('/')[-2]
name = cat_to_name[image_filename]

probs, classes = predict (image_input, model, top_k, power)
probs = probs.data.cpu().numpy().squeeze()
classes = classes.data.cpu().numpy().squeeze()+1


print(probs)
print(name)

# present result
print("v--------------result------------------v")
for i in range(0, len(classes)):
    print("class: {}; with a probability of: {}".format(cat_to_name[str(classes[i])], probs[i]))
print("Λ--------------result------------------Λ")
