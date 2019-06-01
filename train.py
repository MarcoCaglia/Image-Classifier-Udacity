### Import relevant libraries and functions
from transform import image_transformation
from torchvision import models
from torch import nn, optim
from utility import validation,save_model
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',type=str)
parser.add_argument('--save_dir',type=str,default='icp.pth')
parser.add_argument('--arch',default='vgg19',type=str)
parser.add_argument('--gpu',action='store_true',default=False)
parser.add_argument('--learning_rate',default=0.003,type=float)
parser.add_argument('--hidden_units',default=512,type=int)
parser.add_argument('--epochs',default=10,type=int)
args = parser.parse_args()

### Closed: Get images
directory = args.data_dir
train_path = directory + '/train'
val_path = directory + '/valid'
test_path = directory + '/test'
loader = image_transformation(train_path,val_path,test_path)

with open('cat_to_name.json','r') as file:
    categories_to_names = json.load(file)

### Open: Train and save model
# Load model and modify final layer
model_dict = {'vgg19':[models.vgg19(pretrained=True),25088,'classifier'],
              'resnet152':[models.resnet152(pretrained=True),2048,'fc'],
              'squeezenet1_1':[models.squeezenet1_1(pretrained=True),512,'classifier'],
              'densenet161':[models.densenet161(pretrained=True),2208,'classifier']}

model = model_dict[args.arch][0]

for param in model.parameters():
    param.requires_grad=False

if model_dict[args.arch][2] == 'fc':
    model.fc = nn.Sequential(nn.Linear(model_dict[args.arch][1],args.hidden_units),
                         nn.Dropout(p=0.2),
                         nn.ReLU(),
                         nn.Linear(args.hidden_units,500),
                         nn.Dropout(p=0.2),
                         nn.ReLU(),
                         nn.Linear(500,250),
                         nn.Dropout(p=0.2),
                         nn.ReLU(),
                         nn.Linear(250,102),
                         nn.LogSoftmax(dim=1))
    
else:
    model.classifier = nn.Sequential(nn.Linear(model_dict[args.arch][1],args.hidden_units),
                         nn.Dropout(p=0.2),
                         nn.ReLU(),
                         nn.Linear(args.hidden_units,500),
                         nn.Dropout(p=0.2),
                         nn.ReLU(),
                         nn.Linear(500,250),
                         nn.Dropout(p=0.2),
                         nn.ReLU(),
                         nn.Linear(250,102),
                         nn.LogSoftmax(dim=1))

model.cat_to_idx = categories_to_names

# Set training parameters
criterion = nn.NLLLoss()

epochs = args.epochs
if model_dict[args.arch][2] == 'fc':
    optimizer = optim.SGD(model.fc.parameters(),lr=args.learning_rate)
else:
    optimizer = optim.SGD(model.classifier.parameters(),lr=args.learning_rate)

if args.gpu:
    device='cuda'
else:
    device='cpu'

model.to(device)

# Training the model
for epoch in range(epochs):
    train_loss = 0
    step = 0
    for images, labels in loader['trainloader']:
        step +=1
        model.train()
        images, labels =images.to(device),labels.to(device)
        optimizer.zero_grad()
        
        yhat = model(images)
        loss = criterion(yhat,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if step % 30 == 0:
            val_loss,val_acc = validation(model,criterion,loader['valloader'],device)
            print('Step: {}; Validation Loss: {}; Validation Accuracy: {}'.format(step,val_loss,val_acc))
    
    print('Epoch: {}/{}; Training Loss: {}'.format(epoch+1,epochs,train_loss/len(loader['trainloader'])))
          
save_model(model,categories_to_names,args.save_dir)