### Open: Transform pictures into useable format
from torchvision import datasets, models, transforms
import torch
from PIL import Image
import numpy as np

def image_transformation(train_path,val_path,test_path):
    '''Takes in three strings for the training, validation and test data. Applies transformations to the images found in those folders and returns
    one dictionary containing all three torch dataloaders.'''
    
    # Compose the transformations
    train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    
    # Load the datasets
    trainset = datasets.ImageFolder(train_path,transform=train_transforms)
    valset = datasets.ImageFolder(val_path,transform=test_transforms)
    testset = datasets.ImageFolder(test_path,transform=test_transforms)
    
    # Load the images into dictionary
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=True)
    
    loaders = {'trainloader':trainloader,
               'valloader':valloader,
               'testloader':testloader}
    
    # Return loader dictionary
    return loaders

# Open: Single picture transformation function
def process_image(path):
    im = Image.open(path)
    width, height = im.size
    if width >= height:
        width = 256
    else:
        height = 256
    im.thumbnail((width,height))
    im = im.crop((width*0.5-112,height*0.5-112,width*0.5+112,height*0.5+112))
    np_image = np.array(im)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np_image/255
    np_image = (np_image-mean)/std
    np_image = np_image.transpose(2,0,1)
    
    return np_image