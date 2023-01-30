from __future__ import print_function, division
import os,sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

seed=17
np.random.seed(seed)
torch.manual_seed(seed)

class Ising_Dataset(torch.utils.data.Dataset):
    """Ising pytorch dataset."""

    def __init__(self, data_type, transform=False):
        """
        Args:
            data_type (string): `train`, `test` or `critical`: creates data_loader
            transform (callable, optional): Optional transform to be applied on a sample.

        """

        from sklearn.model_selection import train_test_split
        import collections
        import pickle as pickle


        L=40 # linear system size
        T=np.linspace(0.25,4.0,16) # temperatures
        T_c=2.26 # critical temperature in the TD limit

        # load data
        file_name = "Ising_All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
        data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
        data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
        data=data.astype('int')
        data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

        file_name = "Ising_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
        labels = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

        # divide data into ordered, critical and disordered

        X_ordered=data[:70000,:]
        Y_ordered=labels[:70000]

        X_critical=data[70000:100000,:]
        Y_critical=labels[70000:100000]

        X_disordered=data[100000:,:]
        Y_disordered=labels[100000:]

        del data,labels
        # define training, critical and test data sets
        X=np.concatenate((X_ordered,X_disordered)) #np.concatenate((X_ordered,X_critical,X_disordered))
        Y=np.concatenate((Y_ordered,Y_disordered)) #np.concatenate((Y_ordered,Y_critical,Y_disordered))

        # pick random data points from ordered and disordered states to create the training and test sets
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,train_size=0.8)


        if data_type=='train':
            X=X_train
            Y=Y_train
            print("Training on 80 percent of examples")

        if data_type=='test':
            X=X_test
            Y=Y_test
            print("Testing on 20 percent of examples")

        if data_type=='critical':
            X=X_critical
            Y=Y_critical
            print("Predicting on %i critical examples"%len(Y_critical))

        # reshape data back to original 2D-array form
        X=X.reshape(X.shape[0],40,40)

        # these are necessary attributes in dataset class and must be assigned
        self.data=(X,Y)
        self.transform = transform


    # override __len__ and __getitem__ of the Dataset() class

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):

        sample=(self.data[0][idx,...],self.data[1][idx])
        if self.transform:
            sample=self.transform(sample)

        return sample

    
def load_data(kwargs):
    # kwargs:  CUDA arguments, if enabled
    # load and noralise train,test, and data
    train_loader = torch.utils.data.DataLoader(
        Ising_Dataset(data_type='train'),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        Ising_Dataset(data_type='test'),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    critical_loader = torch.utils.data.DataLoader(
        Ising_Dataset(data_type='critical'),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, critical_loader
    
    

class model(nn.Module):
    # create convolutional net
    def __init__(self, N=10, L=40):
        # inherit attributes and methods of nn.Module
        super(model, self).__init__()
        # create convolutional layer with input depth 1 and output depth N
        self.conv1 = nn.Conv2d(1, N, kernel_size=2, padding=1)
        # batch norm layer takes Depth
        self.bn1=nn.BatchNorm2d(N)
        # create fully connected layer after maxpool operation reduced 40->18
        self.fc1 = nn.Linear(20*20*N, 2)
        self.N=N
        self.L=L
        print("The number of neurons in CNN layer is %i"%(N))

    def forward(self, x):
        #Unsqueeze command indicates one channel and turns x.shape from (:,40,40) to (:,1, 40,40)
        x=F.relu(self.conv1(torch.unsqueeze(x,1).float()))
        #print(x.shape)  often useful to look at shapes for debugging
        x = F.max_pool2d(x,2)
        #print(x.shape)
        x=self.bn1(x) # largely unnecessary and here just for pedagogical purposes
        return F.log_softmax(self.fc1(x.view(-1,20*20*self.N)), dim=1)

def train(epoch):
    CNN.train() # effects Dropout and BatchNorm layers
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = CNN(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(data_loader,verbose='Test'):
    CNN.eval() # effects Dropout and BatchNorm layers
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = CNN(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(data_loader.dataset)
    print('\n'+verbose+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    accuracy=100. * correct / len(data_loader.dataset)
    return(accuracy)
    
import argparse # handles arguments
#import sys; sys.argv=['']; del sys # required to use parser in jupyter notebooks

# training settings
parser = argparse.ArgumentParser(description='PyTorch Convmodel Ising Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.epochs=5
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

cuda_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

import torch.nn.functional as F # implements forward and backward definitions of an autograd operation
import torch.optim as optim # different update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc

# load data
train_loader, test_loader, critical_loader=load_data(cuda_kwargs)

test_array=[]
critical_array=[]

# create array of depth of convolutional layer
N_array=[1,5,10,20,50]

# loop over depths
for N in N_array:
    CNN = model(N=N)
    if args.cuda:
        CNN.cuda()

    # negative log-likelihood (nll) loss for training: takes class labels NOT one-hot vectors!
    criterion = F.nll_loss
    # define SGD optimizer
    optimizer = optim.SGD(CNN.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(DNN.parameters(), lr=0.001, betas=(0.9, 0.999))

    # train the CNN and test its performance at each epoch
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch==args.epochs:
            test_array.append(test(test_loader,verbose='Test'))
            critical_array.append(test(critical_loader,verbose='Critical'))
        else:
            test(test_loader,verbose='Test')
            test(critical_loader,verbose='Critical')
    print(test_array)
    print(critical_array)

#from matplotlib import pyplot as plt
#
### Print the result for different N
#%matplotlib inline
#
#plt.plot(N_array, test_array, 'r-*', label="test")
#plt.plot(N_array, critical_array, 'b-s', label="critical")
#plt.ylim(60,110)
#plt.xlabel('Depth of hidden layer', fontsize=24)
#plt.xticks(N_array)
#plt.ylabel('Accuracy', fontsize=24)
#plt.legend(loc='best', fontsize=24)
#plt.tick_params(axis='both', which='major', labelsize=24)
#plt.tight_layout()
#plt.show()
