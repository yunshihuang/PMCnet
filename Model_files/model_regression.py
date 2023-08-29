import numpy as np
from torch import nn
from Model_files.base_net import *
from Model_files.priors import *
from Model_files.functions import *

class BayesLinear_Normalq(nn.Module):
    # only one layer  
    """Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """
    def __init__(self, n_in, n_out, prior_W, prior_b, regularization_weight):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_W = prior_W
        self.prior_b = prior_b
        self.regularization_weight = regularization_weight

        # Learnable parameters -> Initialisation is set empirically.
        self.W = nn.Parameter(torch.randn(self.n_in, self.n_out))
        self.b = nn.Parameter(torch.randn(self.n_out))

        self.lpw = 0

    def forward(self, X):
        output = torch.mm(X, self.W) + self.b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)

        if self.prior_W == 'no_prior':
            lpw = torch.tensor(0.0)   
        elif self.prior_W == 'L2':
            lpw = evaluate_regularization(self.regularization_weight, self.W, self.b, self.prior_W)
        else:    
            lpw = self.prior_W.loglike(self.W) + self.prior_b.loglike(self.b)  # Gaussian prior
        return output, lpw



class bayes_linear_2L(nn.Module):
    # all the layers combined together 
    """2 hidden layer Bayes By Backprop (VI) Network"""
    def __init__(self, input_dim, output_dim, n_hid1,n_hid2, prior_W, prior_b, regularization_weight):
        super(bayes_linear_2L, self).__init__()

        self.prior_W = prior_W
        self.prior_b = prior_b

        self.input_dim = input_dim
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.output_dim = output_dim
        
        self.regularization_weight = regularization_weight
        
        if self.n_hid2 == 0:
            self.bfc1 = BayesLinear_Normalq(self.input_dim, self.n_hid1, self.prior_W, self.prior_b, self.regularization_weight)
            self.bfc3 = BayesLinear_Normalq(self.n_hid1, self.output_dim, self.prior_W, self.prior_b, self.regularization_weight)
        else:
            self.bfc1 = BayesLinear_Normalq(self.input_dim, self.n_hid1, self.prior_W, self.prior_b, self.regularization_weight)
            self.bfc2 = BayesLinear_Normalq(self.n_hid1, self.n_hid2, self.prior_W, self.prior_b, self.regularization_weight)
            self.bfc3 = BayesLinear_Normalq(self.n_hid2, self.output_dim, self.prior_W, self.prior_b, self.regularization_weight)

        # choose your non linearity
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        tlpw = 0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, lpw = self.bfc1(x)
        tlpw = tlpw + lpw
        # -----------------
        x = self.act1(x) # ReLU function
        # -----------------
        if self.n_hid2 != 0:
            x, lpw = self.bfc2(x)
            tlpw = tlpw + lpw
            # -----------------
            x = self.act1(x)
        # -----------------
        x, lpw = self.bfc3(x)
        tlpw = tlpw + lpw
        # -----------------

        return x, tlpw

class BBP_Bayes_Net(BaseNet):
    """Full network wrapper for Bayes By Backprop nets with methods for training, prediction and weight prunning"""
    def __init__(self, lr=1e-3, channels_in=1, side_in=3, cuda=True, classes=1, batch_size=10, Nbatches=1,
                 nhid1=16,nhid2=8,prior_W = isotropic_gauss_prior(mu=0, sigma=1), prior_b = isotropic_gauss_prior(mu=0, sigma=1), regularization_weight = 0.01):
        super(BBP_Bayes_Net, self).__init__()
        print('Creating Net for regression!')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.classes = classes
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.prior_W = prior_W
        self.prior_b = prior_b
        self.regularization_weight = regularization_weight
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.side_in = side_in
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(50)
        if self.cuda:
            torch.cuda.manual_seed(50)

        self.model = bayes_linear_2L(input_dim=self.channels_in * self.side_in, output_dim=self.classes, n_hid1=self.nhid1,n_hid2=self.nhid2, prior_W=self.prior_W, prior_b=self.prior_b, regularization_weight = self.regularization_weight)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True
        print('    Total params: %.2f' % self.get_nb_parameters())

    def create_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                                  weight_decay=0)

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        self.optimizer.zero_grad()
        out, tlpw = self.model(x)
        loss_ = nn.MSELoss(reduction='sum')
        mlpdw = loss_(out, y) # the MSE part
        Edkl = tlpw / self.Nbatches # the regularization part

        loss = Edkl + mlpdw
        loss.backward()
        self.optimizer.step()
        
        return Edkl.data, mlpdw.data, loss.data
    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        out, _ = self.model(x)

        loss_ = nn.MSELoss(reduction='sum')
        loss = loss_(out, y)

        return out,loss.data       
