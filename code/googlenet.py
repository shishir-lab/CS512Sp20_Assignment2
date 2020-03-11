# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
from data_loader import get_dataset
from read_data import read_train
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

filename='../data/train.txt'
#Function used previously
train_data = read_train(filename)

x = []
y = []

for word in train_data:
    for i in range(word[0].shape[0]):
        temp=word[0][i]
        temp=temp.reshape((16,8))
        x.append(temp)
        y.append(word[1][i])

xtensor=torch.tensor(x)
xx=xtensor.reshape((25953,1,16,8))      #(num_of_examples, channels, h, w)
#xx=xx[0:1000] #comment this after ;;;;;just for testing
ytensor=torch.tensor(y)
#ytensor=ytensor[0:1000]

class myDataset(Dataset):
    def __init__(self,transform=None):
        self.samples = xx.float()
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  
        sample = self.samples
        if self.transform:
            #self.samples[idx] = self.samples[idx].float()
            #print(self.samples[idx].shape)
            sample = self.transform(self.samples[idx])
        return sample

#upscaling image
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
   
xd=myDataset(transform=train_transform)   
####

train_loader = torch.utils.data.DataLoader(xd, batch_size=41, shuffle=True, num_workers=0)   #change batch size to 41 after test

#device = torch.device("cuda:0")

model = models.googlenet(pretrained=False, progress=True)

###############################modifying first and last layers of net##########
model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(1024, 26)

# defining loss, optimizers
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters())

###defined batches (x batches_x is useless)
batches_x=torch.split(xx,41)        #put 41 after testing
batches_y=torch.split(ytensor,41)

#workaround for a runtime error
model.aux_logits=False

#set mode of model
model.train()

current_loss = 0.0
current_acc = 0

# iterate over the training data for one epoch
x=0
for i, (inputs) in enumerate(train_loader):
    
    # send the input/labels to the GPU
    #inputs = inputs.to(device)
    #labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        # forward
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        loss = loss_function(outputs, batches_y[x].long())

        # backward
        loss.backward()
        optimizer.step()

    # statistics
    current_loss += loss.item() * inputs.size(0)
    current_acc += torch.sum(predictions == batches_y[x])
    x+=1

total_loss = current_loss / len(train_loader.dataset)
total_acc = current_acc.double() / len(train_loader.dataset)

print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))