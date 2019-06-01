import torch
from torch import nn
from torchvision import models
import numpy as np

### Open: Validation function (accuracy and test/train loss)
def validation(model,criterion,loader,device):
    with torch.no_grad():
        val_loss = 0
        accuracy = 0
        correct = 0
        total = 0
        for images, labels in loader:
            model.eval()
            images, labels = images.to(device),labels.to(device)
            
            yhat = model(images)
            loss = criterion(yhat,labels).item()
            val_loss += loss
            
            yhat_p = torch.exp(yhat)
            _, predictions = torch.max(yhat_p.data,1)
            equality = (predictions == labels)
            correct += equality.type(torch.FloatTensor).sum().item()
            total += labels.size(0)
    
    accuracy = correct/total
    val_loss = val_loss/len(loader)
    return val_loss, accuracy

### Closed: Save model function
def save_model(model,mapping,path):
    checkpoint = {'model':model,
                  'model_sd':model.state_dict()}

    torch.save(checkpoint,path)
              
### Close: Load model function
def load_model(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_sd'])
    
    return model

### Open: Use a model to predict the top_k probabilities and the top_k classes
def predict(model,input_tensor,top_k=1,device='cpu'):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    yhat = torch.exp(model(input_tensor.unsqueeze_(0).float())).cpu()
    probs, classes = yhat.topk(top_k,dim=1)
    probs, classes = np.array(probs[0].detach().numpy()), np.array(classes[0].detach().numpy())
    classes = classes.astype('str')
    classes = np.array([model.class_to_idx[i] for i in classes])[probs.argsort()[::-1]]
    probs = np.sort(probs)[::-1]
    
    result = np.column_stack((classes,probs))
    
    return result