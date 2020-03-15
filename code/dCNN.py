# -*- coding: utf-8 -*-

import torch
import torchvision.models as models
from read_data import read_train
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from lenet import LeNet
import numpy as np
device='cuda'
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

train_loader = torch.utils.data.DataLoader(xd, batch_size=41, shuffle=False, num_workers=0)

#batches_x=torch.split(xx,41)        #put 41 after testing
batches_y=torch.split(ytensor,41)

#_________________________________________________________________________________#

filename2='../data/test.txt'
test_data_ = read_train(filename2)

x_ = []
y_ = []

for word_ in test_data_:
    for i in range(word_[0].shape[0]):
        temp=word_[0][i]
        temp=temp.reshape((16,8))
        x_.append(temp)
        y_.append(word_[1][i])

xtensor_=torch.tensor(x_)
xx_=xtensor_.reshape((xtensor_.shape[0],1,16,8))      #(num_of_examples, channels, h, w)
ytensor_=torch.tensor(y_)

class myDataset_(Dataset):
    def __init__(self,transform=None):
        self.samples = xx_.float()
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  
        sample = self.samples
        if self.transform:
            sample = self.transform(self.samples[idx])
        return sample

xtest=myDataset_(transform=train_transform)   
####

test_loader = torch.utils.data.DataLoader(xtest, num_workers=4, batch_size=41)

batches_y_=torch.split(ytensor_,41)

train_letter_accs = []
train_word_accs = []

test_letter_accs = []
test_word_accs = []

model = LeNet()
model.cuda()
epochs=150


loss_func = torch.nn.CrossEntropyLoss()
# SGD used for optimization, momentum update used as parameter update  
optimization = torch.optim.Adam(model.parameters())#, momentum=0.9)
for epoch in range(epochs):  
  model.train()
  current_loss = 0.0
  current_acc = 0
  x=0  
  p_tags=[]   
  for i, (inputs) in enumerate(train_loader):

      with torch.set_grad_enabled(True):
          # forward
          inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(batches_y[x].cuda()) 
          optimization.zero_grad()         
          # forward, backward pass with parameter update
          forward_output = model(inputs)

          loss = loss_func(forward_output, labels)
          loss.backward()   
          optimization.step() 

          predicted_val = forward_output
          predicted_val = predicted_val.cpu()
          _, predicted_val = torch.max(predicted_val, 1)

          pred_label = predicted_val.cpu()
          temp=pred_label.tolist()
          p_tags = p_tags+temp
      
      current_loss += loss.item() * inputs.size(0)
      current_acc += torch.sum(predicted_val == batches_y[x])
      x+=1

  total_loss = current_loss / len(train_loader.dataset)
  #train acc letter
  total_acc = current_acc.double() / len(train_loader.dataset)

  train_letter_accs.append(total_acc)

  #train acc word
  count=0
  start=0
  end=0
  arr_train=[]

  p_label_tests = p_tags

  for word in train_data:

      end = end + word[1].shape[0]
      if np.array_equal(word[1],p_label_tests[start:end]):
          arr_train.append(True)
      else:
          arr_train.append(False)
      start = end
      #count+=1
  whole_word_acc_train = sum(arr_train)/len(arr_train)

  train_word_accs.append(whole_word_acc_train)

    ###letter wise acc for test
  model.eval()
  model.cuda()

  correct_ = []
  p_tags_ = []
  itr=0
  ytensor_.cuda()
  with torch.no_grad():
    for test in test_loader:
      #batches_y[itr].cuda()
      inputs = torch.autograd.Variable(test.cuda())
      inputs.cuda()
      out = model(inputs)
      _, pred_label = torch.max(out, 1)
      pred_label = pred_label.cpu()
      #print(pred_label.device)
      temp_=pred_label.tolist()
      p_tags_ = p_tags_+temp_
      temp2_ = (pred_label == batches_y_[itr]).tolist()
      correct_ = correct_ + temp2_
      itr+=1
      
  corr_=sum(correct_)
  total_=ytensor_.shape[0]
  total_acc_test = corr_/total_

  test_letter_accs.append(total_acc_test)

  #####test word acc############################################
  count=0
  start=0
  end=0
  arr_test=[]

  p_label_tests = p_tags_

  for word_ in test_data_:

      end = end + word_[1].shape[0]
      if np.array_equal(word_[1],p_label_tests[start:end]):
          arr_test.append(True)
      else:
          arr_test.append(False)
      start = end
      #count+=1
  whole_word_acc_test = sum(arr_test)/len(arr_test)

  test_word_accs.append(whole_word_acc_test)
  #print(whole_word_acc_test)

  print("Epoch", epoch+1)
  print('Train Loss: {:.4f}; Accuracy: {:.4f}; Word_Accuracy: {:.4f}'.format(total_loss, total_acc,whole_word_acc_train))
  print('Accuracy: {:.4f}; Word_Accuracy: {:.4f}'.format(total_acc_test, whole_word_acc_test))

import matplotlib.pyplot as plt

iterations = range(1,151)
plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Letter and Word wise Accuracy (Train Data)')
plt.plot(iterations,train_letter_accs,label = "Train Letter Acc")
plt.plot(iterations,train_word_accs, label = "Train Word Acc")
plt.plot(iterations,test_letter_accs, label = "Test Letter Acc")
plt.plot(iterations,test_word_accs, label = "Test Word Acc")
plt.legend()
plt.show()

#plt.figure()
#plt.xlabel('Iterations')
#plt.ylabel('Letter and Word wise Accuracy (Test Data)')

#plt.show()

#train_letter_accs
#train_word_accs

#test_letter_accs
#test_word_accs

train_letter_accs_l = []
train_word_accs_l = []

test_letter_accs_l = []
test_word_accs_l = []

model = LeNet()
model.to(device)
epochs=150
model.train()

loss_func = torch.nn.CrossEntropyLoss()
imp=[]
optimization = torch.optim.LBFGS(model.parameters(),history_size=10, max_iter=4)
for epoch in range(epochs):  
  model.train()
  current_loss = 0.0
  current_acc = 0
  x=0  
  p_tags=[]  
  w=[] 
  lk=[]
  for i, (inputs) in enumerate(train_loader):
      with torch.set_grad_enabled(True):
          # forward
          inputs, labels = torch.autograd.Variable(inputs.to(device)), torch.autograd.Variable(batches_y[x].to(device)) 
          #inputs=inputs.to(device)
          #batches_y=batches_y.to(device)
          #labels=batches_y[x].to(device)
          def closure():
            optimization.zero_grad()         
            # forward, backward pass with parameter update
            forward_output = model(inputs)
            loss = loss_func(forward_output, labels)
            #print(loss.requires_grad)
            loss.backward() 

            lk.append(loss)
            print(epoch)
            print(loss)
            
            w.append(model.fc3.weight.grad)
            #print(model.fc3.weight.grad)
            return loss
          optimization.step(closure)
      x+=1
  ###letter wise acc for test
  #model.eval()
  imp.append(lk)
  model.to(device)

  correct_ = []
  p_tags_ = []
  itr=0
  ytensor_.to(device)
  with torch.no_grad():
    for test in test_loader:
      #batches_y[itr].to(device)
      inputs = torch.autograd.Variable(test.to(device))
      inputs.to(device)
      out = model(inputs)
      _, pred_label = torch.max(out, 1)
      pred_label = pred_label.cpu()
      #print(pred_label.device)
      temp_=pred_label.tolist()
      p_tags_ = p_tags_+temp_
      temp2_ = (pred_label == batches_y_[itr]).tolist()
      correct_ = correct_ + temp2_
      itr+=1
      
  corr_=sum(correct_)
  total_=ytensor_.shape[0]
  total_acc_test = corr_/total_

  test_letter_accs_l.append(total_acc_test)

  #####test word acc############################################
  count=0
  start=0
  end=0
  arr_test=[]

  p_label_tests = p_tags_

  for word_ in test_data_:

      end = end + word_[1].shape[0]
      if np.array_equal(word_[1],p_label_tests[start:end]):
          arr_test.append(True)
      else:
          arr_test.append(False)
      start = end
      #count+=1
  whole_word_acc_test = sum(arr_test)/len(arr_test)

  test_word_accs_l.append(whole_word_acc_test)
  #print(whole_word_acc_test)

  print("Epoch", epoch+1)
  #print('Train Loss: {:.4f}; Accuracy: {:.4f}; Word_Accuracy: {:.4f}'.format(total_loss, total_acc,whole_word_acc_train))
  print('Accuracy: {:.4f}; Word_Accuracy: {:.4f}'.format(total_acc_test, whole_word_acc_test))

iterations = range(1,151)
plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Letter and Word wise Accuracy (Test Data)')
plt.plot(iterations,train_letter_accs_l,label = "Train Letter Acc")
plt.plot(iterations,train_word_accs_l, label = "Train Word Acc")
plt.plot(iterations,test_letter_accs_l, label = "Test Letter Acc")
plt.plot(iterations,test_word_accs_l, label = "Test Word Acc")
plt.legend()
plt.show()

#plt.figure()
#plt.xlabel('Iterations')
#plt.ylabel('Letter and Word wise Accuracy (Test Data)')
