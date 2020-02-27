# Image Classifier Part-2: Training a CNN Model
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

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="/home/workspace/ImageClassifier/flowers", help="path to training data")
parser.add_argument("--gpu", default="gpu", help="trains on GPU", action="store")
parser.add_argument("--structure", default= "vgg16", help="structure")
parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
parser.add_argument("--dropout", default=0.4, type=float)
parser.add_argument("--epochs", default=3, type= int, help="Number of Epochs")
parser.add_argument("--hidden_units", default=4096, type =int, help="Hidden units")
parser.add_argument("--save_dir", default="/home/workspace/ImageClassifier/checkpoint.pth", help="path and filename")
args = parser.parse_args()
                    
# DONE: GPU 

if args.gpu == "gpu":
    power = "cuda"
else:
    power = "cpu"

 # DONE: Parser
                    
data_dir = args.data_dir
learning_rate = args.learning_rate
hidden_units = args.hidden_units
dropout = args.dropout
epochs = args.epochs
structure = args.structure
save_dir =args.save_dir

#DONE: Print settings

print(power)                    
print(structure)
print(learning_rate)
print(dropout)
print(hidden_units)
print(epochs)
print(save_dir)

                                                        
print("Training Input from:", data_dir)

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
                    
# DONE: Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomCrop (224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
                                     
validate_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# DONE: Load the datasets with ImageFolder


train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
validate_data = datasets.ImageFolder(valid_dir, transform = validate_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# DONE: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(validate_data, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)

classes = train_data.classes

print ('n classes in trainloader:',len(trainloader))
print ('n of pics:', len(trainloader.dataset))                    
                    
# DONE: Define modell structure

structures = {"vgg16":25088,
              "densenet121" : 1024,
              "alexnet" : 9216 }                    
                    
# DONE: Eingangsgrößen und Modell Auswahl
                    
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
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
     
        
        return model , optimizer ,criterion 

    
model,optimizer,criterion = nn_setup(structure, dropout, hidden_units, learning_rate)

#DONE: Training of CNN with Training Data
                  
def train_network(epochs = 8, print_every = 12, power = "cuda"):
                    
    steps = 0
    loss_show=[]


    #DONE : train network model

   # Modell to power
    model.to(power)

    print("Starte Training mit Anzahl Epochen:", epochs)
        # validieren

    for epoch in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs,labels = inputs.to(power), labels.to(power)

            optimizer.zero_grad()


            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vloss = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(validloader):
                    optimizer.zero_grad()

                    inputs2, labels2 = inputs2.to(power) , labels2.to(power)
                    model.to(power)
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vloss = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()


                vloss = vloss / len(validloader)
                accuracy = accuracy*100/len(validloader)

                print('Epoch: {}/{}...'.format(epoch+1, epochs),
                      'Loss: {:.3f}...'.format(running_loss/print_every),
                      'Validation loss: {:.3f}...'.format(vloss),
                      'Validation accuracy: {:.3f}..'.format(accuracy))


                running_loss = 0


   
    print("-------------finished training--------------")
   
                    
train_network(epochs,12,power) 

# DONE: Validation on the test set

def validation(testloader, power = "cuda"): 
     
    print ("Starte Validation") 
                    
    correct = 0
    total = 0
                    
    model.to(power)
                    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(power), labels.to(power)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                    
    print("-------------finished validating--------------")
    print('Test Accuracy: %d %%' % (100 * correct / total))
    
validation(testloader, power) 

# DONE: save model 

model.class_to_idx = train_data.class_to_idx
model.cpu

torch.save({'structure' :'vgg16',
            'hidden_units':4096,
            'epochs': epochs,
            'classifier' : model.classifier,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx},
            save_dir)

print("model is saved to:", save_dir)

              