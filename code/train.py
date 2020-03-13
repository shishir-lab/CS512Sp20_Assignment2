#%%
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF
from crf_utils import compare
import time
import pprint
pp = pprint.PrettyPrinter(indent=4)
#%%

cuda = torch.cuda.is_available()

##################################################
# Begin training
##################################################

# Fetch dataset
dataset = get_dataset()
split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).double(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).double(), torch.tensor(test_target).long())

# Convert to torch
test_X = torch.from_numpy(test_data).double()
test_Y = torch.from_numpy(test_target).long()

if cuda:
    test_X = test_X.cuda()
    test_Y = test_Y.cuda()

#%%
# Model parameters
conv_layers = [[5, 2, (2,1)], [3,1,(1,1)]] #
input_dim = (16,8)
num_labels = 26
# Instantiate the CRF model
testcrf = CRF(input_dim=(16,8), conv_layers=conv_layers, num_labels=26, C=1000)
print(testcrf)

# Setup the optimizer
opt = optim.LBFGS(testcrf.parameters(), max_iter=5, lr=1)

# Tunable parameters
step = 0
batch_size = 64
num_epochs = 10
max_iters  = 100
print_iter = 1 # Prints results every n iterations
log = []
#%%
for i in range(num_epochs):
    print("Processing epoch {}".format(i))

    # Define train and test loaders
    train_loader = data_utils.DataLoader(train,  # dataset to load from
                                          batch_size=batch_size,  # examples per batch (default: 1)
                                          shuffle=True,
                                          sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                          num_workers=0,  # subprocesses to use for sampling
                                          pin_memory=False,  # whether to return an item pinned to GPU
                                          )

    test_loader = data_utils.DataLoader(test,  # dataset to load from
                                        batch_size=batch_size,  # examples per batch (default: 1)
                                        shuffle=False,
                                        sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        num_workers=0,  # subprocesses to use for sampling
                                        pin_memory=False,  # whether to return an item pinned to GPU
                                        )
    print('Loaded dataset... ')

    # Now start training
    for i_batch, sample in enumerate(train_loader):

        train_X = sample[0]
        train_Y = sample[1]

        if cuda:
            train_X = train_X.cuda()
            train_Y = train_Y.cuda()
        # compute loss, grads, updates:
        def closure():
            opt.zero_grad() # clear the gradients
            tr_loss = testcrf.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
            tr_loss.backward() # Run backward pass and accumulate gradients
            print(tr_loss)
            return tr_loss
            
        opt.step(closure) # Perform optimization step (weight updates)

        # print to stdout occasionally:
        if step % print_iter == 0:
            with torch.no_grad():
                test_loss = testcrf.loss(test_X, test_Y)
                tr_loss = testcrf.loss(train_X, train_Y)
                print(step, tr_loss.data, test_loss.data,
                           tr_loss.data / batch_size, test_loss.data / test_Y.shape[0])
    
			##################################################################
			# IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
			##################################################################
            preds = testcrf(test_X)
            letterAcc, wordAcc = compare(test_X, test_Y, preds)
            trPreds = testcrf(train_X)
            trletterAcc, trwordAcc = compare(train_X, train_Y, trPreds)
            t = time.time()
            log.append({
                "step": step,
                "time": t,
                "batchSize": batch_size,
                "testSize": test_Y.shape[0],
                "trainLetterAcc": trletterAcc.item(),
                "trainWordAcc": trwordAcc,
                "testLetterAcc": letterAcc.item(),
                "testWordAcc": wordAcc,
                "trainLoss": tr_loss.item(),
                "testLoss": test_loss.item()
                })
            pp.pprint(log[step])

        step += 1
        if step > max_iters: 
            print("Reached Max iters..")
            # break
            raise StopIteration
    # del train, test
                    
#%%            
import matplotlib.pyplot as plt

if cuda:
    machine = "GPU"
else:
    machine = "CPU"

wallClock = np.array([x["time"] for x in log])
wallClock = wallClock - np.min(wallClock)
testLetterAcc = np.array([x["testLetterAcc"] for x in log])
testWordAcc = np.array([x["testWordAcc"] for x in log])
trainLetterAcc = np.array([x["trainLetterAcc"] for x in log])
trainWordAcc = np.array([x["trainWordAcc"] for x in log])

plt.plot(wallClock, trainLetterAcc, label="Train Letter Accuracy", color='blue')
plt.plot(wallClock, trainWordAcc, label="Train Word Accuracy", color='lightblue')
plt.plot(wallClock, testLetterAcc, label="Test Letter Accuracy", color='green')
plt.plot(wallClock, testWordAcc, label="Test Word Accuracy", color='lightgreen')
plt.grid()
plt.legend()
plt.xlabel("Relative Wall Clock (with {})".format(machine))
plt.ylabel("Accuracy")
plt.title("Minibatch Training with C=1000, LBFGS Iteration=5, ConvLayer=[[5,2,(2,1)], [3,1,(1,1)]]")
plt.show()
#%%
from torchviz import make_dot, make_dot_from_trace
loss = testcrf.loss(train_X, train_Y)
print(loss)
g = make_dot(loss, params=dict(testcrf.named_parameters()))
g.render('../result/computation_graph_convcrf.gv', view=True)  