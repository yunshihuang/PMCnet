import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
import math
import sys
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from sklearn import metrics
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy.io as sio
import time
import pandas as pd
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pickle
from functions_new import *
from algo_new import *
from model_regression import *
from golden_search import *
from run_BNN_fixedReg_regression_large import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import traceback

cuda2 = torch.device('cuda:0')

folder_name_train = 'data/Naval/train_data.mat'
folder_name_val = 'data/Naval/val_data.mat'
folder_name_test = 'data/Naval/test_data.mat'

y_train = OpenMat(sio.loadmat(folder_name_train)['y1_train'])
x_train1 = OpenMat(sio.loadmat(folder_name_train)['x_train'])
y_val = OpenMat(sio.loadmat(folder_name_val)['y1_val'])
x_val1 = OpenMat(sio.loadmat(folder_name_val)['x_val'])
y_test = OpenMat(sio.loadmat(folder_name_test)['y1_test'])
x_test1 = OpenMat(sio.loadmat(folder_name_test)['x_test'])

x_train = x_train1
x_val = x_val1
x_test = x_test1
#scale = MinMaxScaler(feature_range = (0,1))
scale = StandardScaler()
x_train = torch.Tensor(scale.fit_transform(x_train1))
x_val = torch.Tensor(scale.fit_transform(x_val1))
x_test = torch.Tensor(scale.fit_transform(x_test1))

# x_train = x_train) # transform to torch tensor
y_train = torch.Tensor(y_train).squeeze(0).unsqueeze(-1)
# x_val = torch.Tensor(x_val) # transform to torch tensor
y_val = torch.Tensor(y_val).squeeze(0).unsqueeze(-1)
# x_test = torch.Tensor(x_test) # transform to torch tensor
y_test = torch.Tensor(y_test).squeeze(0).unsqueeze(-1)

print('The size of x_train is ', x_train.shape)
print('The size of y_train is ', y_train.shape)
print('The size of x_val is ', x_val.shape)
print('The size of y_val is ', y_val.shape)
print('The size of x_test is ', x_test.shape)
print('The size of y_test is ', y_test.shape)

batch_size = 100

train_dataset = TensorDataset(x_train,y_train) # create your datset
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 


val_dataset = TensorDataset(x_val,y_val) # create your datset
val_loader = DataLoader(val_dataset, batch_size = x_val.shape[0], shuffle = True) # not use minibatch


test_dataset = TensorDataset(x_test,y_test) # create your datset
test_loader = DataLoader(test_dataset, batch_size =  batch_size, shuffle = False) # not use minibatch

# create a dictionary to save options
hidden_layer1 = 100 # the number of neurons in the hidden layer # L=2
hidden_layer2 = 100
tp = {}
tp['L'] = 3
tp['S'] = 1
if tp['L'] != 1:
    tp['M'] = [x_train.shape[1],hidden_layer1,hidden_layer2,1] # the map
else:
    tp['M'] = [x_train.shape[1],1] # the map
tp['prior'] = 'L2' #'no_prior', 'Gaussian_prior', 'Laplace_prior','L2'
tp['x_0'] = x_train
tp['y'] = y_train
tp['regularization_weight'] = 1.480264670576135
if tp['prior'] == 'no_prior':# MLE
    prior_W = 'no_prior'
    prior_b = 'no_prior'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'Gaussian_prior':
    prior_W = isotropic_gauss_prior(mu=0, sigma=2)
    prior_b = isotropic_gauss_prior(mu=0, sigma=2)
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'Laplace_prior':# MAP+L1 regularization
    prior_sig = 0.1
    prior = laplace_prior(mu=0, b=prior_sig)
elif tp['prior'] == 'L2': # L2 regularization
    prior_W = 'L2'
    prior_b = 'L2'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'L1': # L1 regularization
    prior_W = 'L1'
    prior_b = 'L1'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b       
print('The prior is ',tp['prior'])        
L = tp['L']
M = tp['M']
total_number = 0
for ll in range(L):
    total_number += M[ll]*M[ll+1]+M[ll+1]
tp['dimension'] = total_number  
tp['activation'] = ['ReLU','ReLU','None']
tp['classification'] = 'regression'
dogolden_search = 0
dosave = 1

# some settings
use_cuda = torch.cuda.is_available()
lr = 1e-3
models_dir = 'BNN_models/Naval/BNN_Gaussian'
results_dir = 'BNN_results/Naval/BNN_Gaussian'  

# We do the training using ADAM without regularization first
net = BBP_Bayes_Net(lr=lr, channels_in=1, side_in=x_train.shape[1], cuda=use_cuda, classes=1, batch_size=batch_size,
                        Nbatches=(x_train.shape[0]/batch_size), nhid1=hidden_layer1,nhid2=hidden_layer2,
                        prior_W=prior_W,prior_b=prior_b,regularization_weight=tp['regularization_weight'])

torch.set_printoptions(precision=100)
np.set_printoptions(precision=100)

# load the trained model
path_save         = os.path.join(models_dir,'trained_model_MinLossOnVal_map_test.pt')
net.model.load_state_dict(torch.load(path_save,map_location=torch.device('cpu')))

# the results
# print the parameters W and b of two 1200 unit ReLU layers
k = 0
no_param = 0
for f in net.model.parameters():
    if k == no_param:
        W1 = f.data
        #print('W1 is',W1)
    elif k == no_param+1:
        #print('This is',k)
        b1 = f.data
    elif k == no_param+2:
        #print('This is',k)
        W2 = f.data
    elif k == no_param+3:
        #print('This is',k)
        b2= f.data
    elif k == no_param+4:
        #print('This is',k)
        W3 = f.data
    elif k == no_param+5:
        #print('This is',k)
        b3= f.data       
    k += 1    

#parameters for our algorithm
p=time.time()
N = 20 # number of proposals
K = 20  # samples per proposal per iteration
sig_prop = 0.0001
lr = 2  #glocal resampling
gr_period=5
tp['regularization_weight'] = 0.2360689191000000#1.480264670576135
epsilon1 = 1e-50
epsilon2 = 1e-50

# est_ml is the set of all parameters (W,b), stacked in a column
est_ml1 = torch.cat((torch.transpose(W1,0,1).reshape(M[0]*M[1],1),b1.reshape(M[1],1)),0)
est_ml2 = torch.cat((torch.transpose(W2,0,1).reshape(M[1]*M[2],1),b2.reshape(M[2],1)),0)
est_ml = torch.cat((est_ml1,est_ml2),0)
est_ml3 = torch.cat((torch.transpose(W3,0,1).reshape(M[2]*M[3],1),b3.reshape(M[3],1)),0)
est_ml = torch.cat((est_ml,est_ml3),0)

print('the shape of est_ml is ',est_ml.shape)

logger = get_logger('log_BNN_naval_l2.txt') 

if dogolden_search == 0:    
    T = 5
    N_resampled = 100
    is_binary = 0
    loss = 'MSE'
    y_train1 = y_train.detach().numpy()
    y_val1 = y_val.detach().numpy()
    y_test1 = y_test.detach().numpy()


    myprint('T is {}'.format(T),logger)
    myprint('regularization_weight is {}'.format(tp['regularization_weight']),logger)
    myprint('sig_prop is {}'.format(sig_prop),logger)
    myprint('N_resampled is {}'.format(N_resampled),logger)
    
    output_vec = []
    ESS_vec = []
    ESS = np.zeros((50,T))
    
        ##This line opens a log file
    with open("bug_log_BNN_multiclass.txt", "w") as log:

        try:
            for i in range(1): 
                myprint('This is simulation {}'.format(i),logger)
                output = SL_PMC_Adapt_Cov_new(test_loader,N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1,epsilon2)

                output_vec.append(output)

                path_save_BNN_output  = os.path.join(results_dir,'output_naval_l2_final.txt')             
                with open(path_save_BNN_output, "wb") as fp:   #Pickling
                    pickle.dump(output_vec, fp)
            
            print("There is no bug.", file = log)
        except Exception:
            traceback.print_exc(file=log)     

else:  
    T = 20
    N_resampled = 100
    is_binary = 1
    loss = 'MSE'
    simulations = 1
    rangereg = [0,1]
    precision = 5e-2
    
    y_train1 = y_train.detach().numpy()
    y_val1 = y_val.detach().numpy()
    y_test1 = y_test.detach().numpy()
    
    myprint('T is {}'.format(T),logger)
    myprint('sig_prop is {}'.format(sig_prop),logger)
    myprint('N_resampled is {}'.format(N_resampled),logger)
    myprint('loss is {}'.format(loss),logger)
    myprint('Simulations is {}'.format(simulations),logger)
    myprint('the range of regularization is {}'.format(rangereg),logger)
    myprint('the precision of golden search is {}'.format(precision),logger)
    myprint('lr is {}'.format(lr),logger)
    myprint('is_binary is {}'.format(is_binary),logger)

    ##This line opens a log file
    with open("bug_log_BNN_regression.txt", "w") as log:

        try:
            crit = lambda reg: run_BNN_fixedReg_regression_large(train_loader,test_loader,simulations,reg,loss,N_resampled,results_dir, N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1,epsilon2,logger)
            reg_final,reg_list,Loss_list = golden_search(crit,rangereg[0],rangereg[1],precision,logger)

            reg1 = np.int(np.round(reg_final,4)*10000)
            if dosave == 1:
                path_save_BNN_output  = os.path.join(results_dir,'final_output_reg_Naval_l2_'+str(reg1)+'.txt')             
                with open(path_save_BNN_output, "wb") as fp:   #Pickling
                    pickle.dump(reg_list, fp)

                path_save_BNN_output  = os.path.join(results_dir,'final_output_loss_Naval_l2_'+str(reg1)+'.txt')           
                with open(path_save_BNN_output, "wb") as fp:   #Pickling
                    pickle.dump(Loss_list, fp)
            myprint('the final regularization weight is {}'.format(reg_final),logger)
            myprint('the reg list is {}'.format(reg_list),logger)  

            print("There is no bug.", file = log)
        except Exception:
            traceback.print_exc(file=log)        

    

