import torch
import torchvision.models as models
from read_data import read_train
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from lenet import LeNet
import numpy as np

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

train_loader = torch.utils.data.DataLoader(xd, shuffle=False, num_workers=0)

#batches_x=torch.split(xx,41)        #put 41 after testing
batches_y=torch.split(ytensor,ytensor.shape[0])

for i,(inp) in enumerate(train_loader):
  ss=inp
  break

model = LeNet()
model.cuda()
loss_func = torch.nn.CrossEntropyLoss() 
optimization = torch.optim.LBFGS(model.parameters(), lr=0.8)

ytensor=ytensor.cuda()
model.cuda()
z=0

for epoch in range(100):
    p_tags=[]
    current_acc=0
    ss=ss.cuda()
    #print("hello")
    def closure():
      optimization.zero_grad()
      forward_output = model(ss)
      #print(forward_output.shape)
      #return
      loss = loss_func(forward_output, ytensor)
      print(epoch)
      print(loss)    
      #print(loss)
      loss.backward()
      #loss.backward()
      return loss
    optimization.step(closure)

    forward_output = model(ss)
    predicted_val = forward_output
    predicted_val = predicted_val.cpu()
    _, predicted_val = torch.max(predicted_val, 1)

    pred_label = predicted_val.cpu()
    temp=pred_label.tolist()
    p_tags = p_tags+temp
    current_acc += sum(pred_label == ytensor.cpu())
    print("acc::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::",current_acc.item()/ytensor.shape[0])
    z+=1
  