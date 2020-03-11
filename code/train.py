#%%
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF
from crf_utils import compare

#%%

# Tunable parameters
batch_size = 64
num_epochs = 10
# max_iters  = 500
print_iter = 5 # Prints results every n iterations
conv_layers = [[5, 2, (2,1)]] #


# Model parameters
input_dim = (16,8)
num_labels = 26
cuda = torch.cuda.is_available()

# Instantiate the CRF model
crf = CRF(input_dim=input_dim, conv_layers=conv_layers, num_labels=num_labels,  C=100)

# Setup the optimizer
opt = optim.LBFGS(crf.parameters(), max_iter=20)


##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()
split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).double(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).double(), torch.tensor(test_target).long())

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
            tr_loss = crf.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
            tr_loss.backward() # Run backward pass and accumulate gradients
            print()
            return tr_loss
            
        opt.step(closure) # Perform optimization step (weight updates)

        # print to stdout occasionally:
        if step % print_iter == 0:
            random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
            test_X = test_data[random_ixs, :]
            test_Y = test_target[random_ixs, :]

            # Convert to torch
            test_X = torch.from_numpy(test_X).double()
            test_Y = torch.from_numpy(test_Y).long()

            if cuda:
                test_X = test_X.cuda()
                test_Y = test_Y.cuda()
            test_loss = crf.loss(test_X, test_Y)
            tr_loss = crf.loss(train_X, train_Y)
            print(step, tr_loss.data, test_loss.data,
                       tr_loss.data / batch_size, test_loss.data / test_Y.shape[0])

			##################################################################
			# IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
			##################################################################
            preds = crf(test_X)
            letterAcc, wordAcc = compare(test_X, test_Y, preds)

        step += 1
        if step > max_iters: 
            print("Reached Max iters..")
            # break
            raise StopIteration
    # del train, test
