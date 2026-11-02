import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import glob
import os
from train_utils import seed_all, get_best_device
from datasets import readfile
import bayes_num_game as bng
import matplotlib.pyplot as plt
DEVICE = get_best_device()

# A simpler set of Bayesian models for the number game. Here, 
# we consider fine-tuning the prior, and also the full model.

x_max = 100 # (numbers between 1 and 100 are allowed)   
x_all = range(0,x_max+1) # include 0 for convenience, although it's not included in any hypothesis

def convert_h_list_to_tensor(h_list):
    assert(len(set(h_list))==len(h_list)), "h_list cannot contain duplicates"
    if isinstance(h_list[0],str): h_list = [int(item) for item in h_list]
    h_tensor = torch.zeros(len(x_all), device=DEVICE)
    h_tensor[h_list] = 1.
    return h_tensor    

def convert_x_list_to_tensor(x_list):
    if isinstance(x_list[0],str): x_list = [int(item) for item in x_list]
    h_tensor = torch.zeros(len(x_all), device=DEVICE)
    for x in x_list:
        h_tensor[x] += 1.
    return h_tensor    

def inv_sigmoid(x): return torch.log(x / (1 - x))

class BayesNumGameNet(nn.Module):

    # Bayesian model behind the number, with all parameter fixed

    def __init__(self, mylambda=bng.mylambda_default, scale_ll_init=30., lapse_rate=0.5):
        # INPUT
        super(BayesNumGameNet, self).__init__()
        assert(isinstance(mylambda,float))
        self.H_all,self.log_prior_init,self.H_names = bng.generate_hypotheses(mylambda)
        self.H_all = torch.from_numpy(np.array(self.H_all)).to(torch.float32)
        self.H_all = self.H_all.to(device=DEVICE) # [nhyp x 101]
        self.log_prior_init = torch.from_numpy(self.log_prior_init).to(torch.float32) # [nhyp]
        self.log_prior_init = self.log_prior_init.to(device=DEVICE)
        self.prior_weights = nn.Parameter(self.log_prior_init, requires_grad=False) # [nhyp]
        self.likelihood_weights = nn.Parameter(scale_ll_init*(self.H_all-1.), requires_grad=False) # [nhyp x 101]
        self.lapse_rate_weight = nn.Parameter(inv_sigmoid(torch.tensor(lapse_rate)),requires_grad=False)
        self.b_rate_weight = nn.Parameter(inv_sigmoid(torch.tensor(0.5)),requires_grad=False)

        # DEBUG
        # assert(torch.allclose(self.log_prior_init,self.log_prior))
        # assert(torch.allclose(self.H_all/self.H_all.sum(dim=1, keepdim=True), torch.exp(self.ll_mat_all)))

    # Re-parameterizations so that we can use unconstrained optimization
    @property 
    def log_prior(self): # [nhyp]
        # Get log-prior 
        return F.log_softmax(self.prior_weights,dim=0)

    @property
    def ll_mat_all(self): # [nhyp x 101]
        # Get log-likelihood
        return F.log_softmax(self.likelihood_weights, dim=1)

    @property
    def lapse_rate(self): # scalar for lapse rate
        return torch.sigmoid(self.lapse_rate_weight)

    def set_lapse_rate(self,lapse_rate):
        self.lapse_rate_weight.data = inv_sigmoid(torch.tensor(lapse_rate, device=DEVICE))
        self.lapse_rate_weight.requires_grad_(False)
        assert(not self.lapse_rate_weight.requires_grad)
    
    @property
    def b_rate(self): # scalar for base rate in lapse model
        return torch.sigmoid(self.b_rate_weight)

    @property
    def H_all_approx(self):  # [nhyp x 101] 
        # Approximate membership function. That is, a relaxed binary matrix for which numbers
        # belong to which hypotheses 
        H = F.softmax(self.likelihood_weights, dim=1) # [nhyp x 101]
        H = H / torch.max(H,dim=1,keepdim=True)[0] # [nhyp x 1]
        return H
   
    def log_posterior(self, data):
        #
        # Compute posterior distribution
        #
        # Input
        #  data : tensor [nobs x 101] 
        # Output
        #  lpost : tensor [nobs x nhyp], log-probability of each hypothesis
        # log_prior = F.log_softmax(self.prior_weights,dim=0) # [nhyp]
        # ll_mat_all = F.log_softmax(self.likelihood_weights, dim=1) # [nhyp x 101]
        ll = torch.matmul(self.ll_mat_all,torch.transpose(data,0,1)) # [nhyp x nobs]        
        lpost = ll + self.log_prior.view(-1,1) # [nhyp x nobs] (for help with broadcasting)
        lpost = torch.transpose(lpost,0,1) # [nobs x nhyp]
        lpost = F.log_softmax(lpost,dim=1)
        return lpost

    def pp(self, data):
        #
        # Posterior predictive distribution
        #
        # Input
        #  data : tensor [nobs x 101]
        # Output
        #  pp : tensor [nobs x 101], probability of each new number belong to the set (NOT LOG)
        lpost = self.log_posterior(data) # [nobs x nhyp]
        post = torch.exp(lpost) # [nobs x nhyp]
        pp = torch.matmul(post,self.H_all) # [nobs x nhyp] x [nhyp x 101] = [nobs x 101]
        return pp

    def forward(self, data):
        # Input
        #  data : tensor [nobs x 101] (count of each number in observation set)
        # Output
        #  out : tensor [nobs x 101 x 2], LOG probability of respond 'yes' [:,:,0] or 'no' to each query number (column)
        pp = self.pp(data) # [nobs x 101]
        pp_smooth = (1-self.lapse_rate)*pp + self.lapse_rate*self.b_rate
        out = torch.stack( (torch.log(pp_smooth), torch.log(1-pp_smooth)),dim=2)
        return out

class BayesTemp(BayesNumGameNet):

    def __init__(self, mylambda=bng.mylambda_default, scale_ll_init=30., lapse_rate=0.5):
        super(BayesTemp, self).__init__(mylambda=mylambda, scale_ll_init=scale_ll_init, lapse_rate=lapse_rate)
        self.temp_posterior = 1.

    def log_posterior(self, data):
        #
        # Compute posterior distribution
        #
        # Input
        #  data : tensor [bsize x 101] 
        # Output
        #  lpost : tensor [bsize x nhyp], log-probability of each hypothesis

        # log_prior # [nhyp]
        # ll_mat_all # [nhyp x 101]
        ll = torch.matmul(self.ll_mat_all,torch.transpose(data,0,1)) # same as original model, [nhyp x bsize]        
        lpost = (1./self.temp_posterior) * (ll + self.log_prior.view(-1,1)) # [nhyp x bsize] 
        lpost = torch.transpose(lpost,0,1) # [bsize x nhyp]
        lpost = F.log_softmax(lpost,dim=1) # [bsize x nhyp]
        assert torch.allclose(torch.sum(torch.exp(lpost), dim=1), torch.tensor(1.0, device=lpost.device)), "exp(lpost) does not sum to 1"
        return lpost

class BayesNumGameOptPrior(BayesNumGameNet):

    # Consider original Bayesian model with fixed hypotheses.clear
    # Here, optimize only their prior probabilities.

    def __init__(self, mode, mylambda=bng.mylambda_default, scale_ll_init=30., lapse_rate=0.5, sigma_init=.001, fit_b_rate=False):
        super(BayesNumGameOptPrior, self).__init__(mylambda=mylambda, scale_ll_init=scale_ll_init, lapse_rate=lapse_rate)
        assert(mode in ['bayes','scratch'])
        if mode=='bayes':
            self.prior_weights = nn.Parameter(self.prior_weights)            
        else:
            self.prior_weights = nn.Parameter( sigma_init*torch.randn(self.prior_weights.size()) ) # [nhyp]
        self.lapse_rate_weight = nn.Parameter(self.lapse_rate_weight)
        self.b_rate_weight = nn.Parameter(self.b_rate_weight, requires_grad=fit_b_rate)
        self.likelihood_weights = self.likelihood_weights.to(device=DEVICE)
        self.reg_strength = float('nan')

class BayesNumGameOptFull(BayesNumGameNet):

    # Consider original Bayesian model.
    # Here, optimize both the hypotheses themselves and the prior probabilities.
    # Note that the posterior predictive function must be modified to do sampling rather than binary membership.

    def __init__(self, mode, mylambda=bng.mylambda_default, scale_ll_init=30., lapse_rate=0.5, sigma_init=3., fit_b_rate=False):
        super(BayesNumGameOptFull, self).__init__(mylambda=mylambda, scale_ll_init=scale_ll_init, lapse_rate=lapse_rate)
        assert(mode in ['bayes','scratch'])
        if mode=='bayes':
            self.likelihood_weights = nn.Parameter(self.likelihood_weights)
            self.prior_weights = nn.Parameter(self.prior_weights)
        elif mode=='scratch':
            self.likelihood_weights = nn.Parameter( sigma_init*torch.randn(self.likelihood_weights.size()) ) # [nhyp x 101]
            self.prior_weights = nn.Parameter( sigma_init*torch.randn(self.prior_weights.size()) ) # [nhyp]
        self.lapse_rate_weight = nn.Parameter(self.lapse_rate_weight)
        self.b_rate_weight = nn.Parameter(self.b_rate_weight, requires_grad=fit_b_rate)
        self.reg_strength = float('nan')

    def pp(self, data):
        #
        # Posterior predictive distribution
        #
        # Input
        #  data : tensor [nobs x 101]
        # Output
        #  pp : tensor [nobs x 101], probability of each new number belong to the set (NOT LOG)
        # H_all_approx = F.softmax(self.likelihood_weights, dim=1) # [nhyp x 101]
        lpost = self.log_posterior(data) # [nobs x nhyp]
        post = torch.exp(lpost) # [nobs x nhyp]
        pp = torch.matmul(post,self.H_all_approx) # [nobs x nhyp] x [nhyp x 101] = [nobs x 101]
        return pp

class BayesNumGameOptFullTemp(BayesNumGameOptFull):

    def __init__(self, mode, mylambda=bng.mylambda_default, scale_ll_init=30., lapse_rate=0.5, sigma_init=3., fit_b_rate=False):
        super(BayesNumGameOptFullTemp, self).__init__(mode, 
            mylambda=mylambda, scale_ll_init=scale_ll_init, lapse_rate=lapse_rate, sigma_init=sigma_init, fit_b_rate=fit_b_rate)
        self.temp_ll_weight = nn.Parameter(torch.tensor(0.))
        self.temp_lprior_weight = nn.Parameter(torch.tensor(0.))

    def log_posterior(self, data):
        #
        # Compute posterior distribution
        #
        # Input
        #  data : tensor [bsize x 101] 
        # Output
        #  lpost : tensor [bsize x nhyp], log-probability of each hypothesis

        # log_prior # [nhyp]
        # ll_mat_all # [nhyp x 101]
        ll = torch.matmul(self.ll_mat_all,torch.transpose(data,0,1)) # same as original model, [nhyp x bsize]        
        lpost = (1./self.temp_ll) * ll + (1./self.temp_lprior) * self.log_prior.view(-1,1) # [nhyp x bsize] 
            # adds temperature parameters
        lpost = torch.transpose(lpost,0,1) # [bsize x nhyp]
        lpost = F.log_softmax(lpost,dim=1) # [bsize x nhyp]
        assert torch.allclose(torch.sum(torch.exp(lpost), dim=1), torch.tensor(1.0, device=lpost.device)), "exp(lpost) does not sum to 1"
        return lpost

    @property
    def temp_ll(self):
        return torch.exp(self.temp_ll_weight)
    
    @property
    def temp_lprior(self):
        return torch.exp(self.temp_lprior_weight)


class BayesNumGameOptExtend(BayesNumGameNet):

    # Consider original Bayesian model with fixed hypotheses.
    # Here, add fresh/new hypotheses and optimize them, with all other fixed.
    # Note that the posterior predictive function must be modified to do sampling rather than binary membership.
    #
    # Input
    #  n_new_hyp : number of new hypotheses we want to add
    #  init_prior_weights : initial weights on prior probabilities (default=[] is tenenbaum model)
    def __init__(self, n_new_hyp, init_prior_weights=[], mylambda=bng.mylambda_default, scale_ll_init=30., lapse_rate=0.5, sigma_init=.1, fit_b_rate=False):
        nn.Module.__init__(self)

        # modification from constructor of base class
        self.H_all,self.prior_weights_fixed,self.H_names = bng.generate_hypotheses(mylambda)
        self.H_all = torch.from_numpy(np.array(self.H_all)).to(torch.float32)
        self.H_all = self.H_all.to(device=DEVICE) # [nhyp x 101]
        self.prior_weights_fixed = torch.from_numpy(self.prior_weights_fixed).to(torch.float32) # [nhyp]
        if isinstance(init_prior_weights,torch.Tensor):
            self.prior_weights_fixed = init_prior_weights.detach() # [nhyp]
        self.likelihood_weights_fixed = scale_ll_init*(self.H_all-1.) # [nhyp x 101]
        self.lapse_rate_weight = inv_sigmoid(torch.tensor(lapse_rate))
        self.b_rate_weight = inv_sigmoid(torch.tensor(0.5))

        # new parameters
        self.n_new_hyp = n_new_hyp 
        self.likelihood_weights_fixed = nn.Parameter(self.likelihood_weights_fixed,requires_grad=False)
        self.prior_weights_fixed = nn.Parameter(self.prior_weights_fixed,requires_grad=False)
        size_dat_space = self.likelihood_weights_fixed.shape[1]
        mean_lw = torch.ones(1,size_dat_space, device=DEVICE)
        mean_lw[:,0] = -30. # integer 0 which is never observed
        mean_pw = torch.max(self.prior_weights_fixed,dim=0,keepdim=True)[0]-1.
        if n_new_hyp==1:
            self.likelihood_weights_extend = nn.Parameter( mean_lw )
            self.prior_weights_extend = nn.Parameter( mean_pw )
        else:
            self.likelihood_weights_extend = nn.Parameter( -sigma_init*torch.randn(n_new_hyp,size_dat_space, device=DEVICE) + mean_lw.to(DEVICE))
            self.prior_weights_extend = nn.Parameter( -sigma_init*torch.randn(n_new_hyp, device=DEVICE) + mean_pw.to(DEVICE) )
        self.lapse_rate_weight = nn.Parameter(self.lapse_rate_weight)
        self.b_rate_weight = nn.Parameter(self.b_rate_weight, requires_grad=fit_b_rate)

    @property
    def likelihood_weights(self):
        return torch.cat((self.likelihood_weights_fixed,self.likelihood_weights_extend),dim=0)

    @property
    def prior_weights(self):
        return torch.cat((self.prior_weights_fixed,self.prior_weights_extend),dim=0)

    def pp(self, data):
        #
        # Posterior predictive distribution
        #
        # Input
        #  data : tensor [nobs x 101]
        # Output
        #  pp : tensor [nobs x 101], probability of each new number belong to the set (NOT LOG)
        # H_all_approx = F.softmax(self.likelihood_weights, dim=1) # [nhyp x 101]
        lpost = self.log_posterior(data) # [nobs x nhyp]
        post = torch.exp(lpost) # [nobs x nhyp]
        pp = torch.matmul(post,self.H_all_approx) # [nobs x nhyp] x [nhyp x 101] = [nobs x 101]
        return pp

def flatten_nested_list(mylist):
    # if each item in list is a len=1 list, remove the encasing list for that item
    for idx,item in enumerate(mylist):
        assert(isinstance(item,list) and len(item)==1), 'each item in list must be len=1 list'
        mylist[idx] = item[0]
    return mylist

def load_human_data(mysplit,mydir='data/num_game',size_dat_space=101):
    # Load human behavioral data from file
    #
    #  Input
    #   mysplit is either  ['train','test','val']
    #  Output
    #   data : [n x 101], n unique support sets (rows) shown to participants. 1 indicates number is present
    #   data_eval : [n x 101 x 2], count-based (0...k) of participant responses, such that [:,:,0] are 'yes' counts and [:,:,1] are 'no' counts
    assert(mysplit in ['train','test','val'])
    mydir = os.path.join(mydir,'human_'+mysplit)
    list_files = glob.glob(mydir+"/*.txt") # all episode files
    list_S = []
    dict_S = {}
    for fn in list_files:
        S = readfile(fn)
        myhash = str(S['xs']) # signature of support set
        if myhash not in dict_S:
            dict_S[myhash] = []
        dict_S[myhash].append(S)
    myhash_unique = sorted([myhash for myhash in dict_S.keys()])
    n = len(myhash_unique)
    data = torch.zeros((n,size_dat_space), device=DEVICE)
    data_eval = torch.zeros((n,size_dat_space,2), device=DEVICE)
        # in last dimension, 'yes' is [0] and 'no' is [1]
    for i,myhash in enumerate(myhash_unique):
        xs_flat = flatten_nested_list(dict_S[myhash][0]['xs'])
        data[i] = convert_h_list_to_tensor(xs_flat)
        xq_flat = []
        yq_flat = []
        for S in dict_S[myhash]:
            xq_flat += flatten_nested_list(S['xq'])
            yq_flat += flatten_nested_list(S['yq'])
        for qq in range(len(xq_flat)): # for each possible query number
            if yq_flat[qq] == 'yes': # add pos response
                data_eval[i,:,0] +=  convert_h_list_to_tensor([xq_flat[qq]])
            elif yq_flat[qq] == 'no': # add neg response
                data_eval[i,:,1] +=  convert_h_list_to_tensor([xq_flat[qq]])
            else:
                assert False, "y value should be 'yes' or 'no'"
    data = data.to(device=DEVICE)
    data_eval = data_eval.to(device=DEVICE)
    return data, data_eval

def describe_model(net):
    nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(str(type(net)),'; nparams=',nparams)


def loss_fn_bce(Y_pred_log,data_eval,reduce='mean'):
    # Input
    #  Y_pred_log   [n x 101 x 2] : log-probability of response to each number query (dim=1)
    #  data_eval : [n x 101 x 2], count-based (0...k) of participant responses, such that [:,:,0] is 'yes' and [:,:,1] is 'no'
    assert(reduce in ['mean','none'])
    assert(torch.sum(data_eval[:,0])==0.), "should not have '0' as query"
    loss = -torch.sum(data_eval * Y_pred_log)
    if reduce=='mean':
        loss = loss / torch.sum(data_eval)
    else:
        pass
        # print('nobs=',torch.sum(data_eval).item())
    return loss

def prior_entropy(net):
    # Return : the entropy of the model's prior distribution, used as regularizer
    lp = net.log_prior
    return -torch.sum(torch.exp(lp)*lp)

def fit_lbfgs(net, data, data_eval, nsteps=20, step_tol=1e-6, reg_strength = 1e-3):
    # We want more sophisticated optimizer since we have exact gradient.
    # Input
    #   net : to optimize
    #   data : [n x 101], m unique support sets shown to participants. 1 indicates number is present
    #   data_eval : [n x 101 x 2], participant responses, such that [:,:,0] is 'yes' and [:,:,1] is 'no'
    #   reg_strength : strength of regularization
    # Output
    #   net : optimized net
    def closure():
        optimizer.zero_grad()
        Y_pred_log = net(data) # [n x 101], log-probs
        if reg_strength>0:
            loss = loss_fn_bce(Y_pred_log,data_eval) - reg_strength*prior_entropy(net)
        else:
            loss = loss_fn_bce(Y_pred_log,data_eval)
        loss.backward()
        return loss

    print('Optimizing with LBFGS',end='')
    describe_model(net)
    net.reg_strength = reg_strength
    optimizer = optim.LBFGS(net.parameters(), lr=1., max_iter=200, line_search_fn='strong_wolfe')
    net.train()
    prev_loss = float('inf')
    for step in range(nsteps):        
        optimizer.step(closure)
        myloss = closure().cpu().item()
        print('after step',step,'loss',myloss)
        if not (myloss + step_tol < prev_loss):
            print(' search stopped due to no improvement')
            break
        prev_loss = myloss

def fit_dual_lbfgs(net, data, data_eval, nsteps=20, step_tol=1e-6, reg_strength = 1e-3):
    # Iterate between two optimizers for model of class BayesNumGameOptFullTemp
    #  optimizer 1: handles the Bayes model parameters
    #  optimizer 2: handles the temperature parameters
    # Input
    #   net : to optimize
    #   data : [n x 101], m unique support sets shown to participants. 1 indicates number is present
    #   data_eval : [n x 101 x 2], participant responses, such that [:,:,0] is 'yes' and [:,:,1] is 'no'
    #   reg_strength : strength of regularization
    # Output
    #   net : optimized net
    assert(isinstance(net,BayesNumGameOptFullTemp))
    def closure():
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        Y_pred_log = net(data) # [n x 101], log-probs
        if reg_strength>0:
            loss = loss_fn_bce(Y_pred_log,data_eval) - reg_strength*prior_entropy(net)
        else:
            loss = loss_fn_bce(Y_pred_log,data_eval)
        loss.backward()
        return loss

    print('Optimizing with dual LBFGS',end='')
    describe_model(net)
    net.reg_strength = reg_strength
    print('  parameters to optimize:')
    print([name for name, p in net.named_parameters() if name not in ['temp_ll_weight', 'temp_lprior_weight']])
    params1 = [p for name, p in net.named_parameters() if name not in ['temp_ll_weight', 'temp_lprior_weight']]
    params2 = [net.temp_ll_weight, net.temp_lprior_weight]
    optimizer1 = optim.LBFGS(params1, lr=1., max_iter=200, line_search_fn='strong_wolfe')
    optimizer2 = optim.LBFGS(params2, lr=.05, max_iter=200, line_search_fn='strong_wolfe')
    net.train()
    prev_loss = float('inf')
    for step in range(nsteps):        
        optimizer1.step(closure)
        optimizer2.step(closure)
        myloss = closure().cpu().item()
        print('after step',step,'loss',myloss)
        if not (myloss + step_tol < prev_loss):
            print(' search stopped due to no improvement')
            break
        prev_loss = myloss

def fit_adamw(net, data, data_eval, nsteps=500, reg_strength = 1e-3):
    # AdamW optimizer even though we have the exact gradient.
    # Input
    #   net : to optimize
    #   data : [n x 101], m unique support sets shown to participants. 1 indicates number is present
    #   data_eval : [n x 101 x 2], participant responses, such that [:,:,0] is 'yes' and [:,:,1] is 'no'
    #   reg_strength : strength of regularization
    # Output
    #   net : optimized net
    print('Optimizing with AdamW',end='')
    describe_model(net)
    net.reg_strength = reg_strength
    optimizer = optim.AdamW(net.parameters(), lr=0.0005, betas=(0.9,0.95), weight_decay=0.01)
    net.train()
    prev_loss = float('inf')
    for step in range(nsteps):
        optimizer.zero_grad()
        Y_pred_log = net(data) # [n x 101], log-probs
        if reg_strength>0:
            loss = loss_fn_bce(Y_pred_log,data_eval) - reg_strength*prior_entropy(net)
        else:
            loss = loss_fn_bce(Y_pred_log,data_eval)
        loss.backward()
        optimizer.step()
        if step%20==0: print('step',step,'; loss',loss.item())

def eval(net, data, data_eval, reduce='mean'):
    net.eval()
    Y_pred_log = net(data) # [n x 101], log-probs
    loss = loss_fn_bce(Y_pred_log,data_eval,reduce=reduce)
    return loss.item()

def save_model(fn_out_model, net):
    torch.save(net.state_dict(), fn_out_model)

def load_model(fn_out_model, net):
    nets_state_dict = torch.load(fn_out_model, map_location=DEVICE)
    net.load_state_dict(nets_state_dict)
    net = net.to(device=DEVICE)

def debug():
    BP = BayesNumGameNet()
    # BP = BayesNumGameOptPrior('bayes')
    mypred1 = BP.pp(convert_x_list_to_tensor([16]).view(1,-1))
    mypred2 = BP.pp(convert_x_list_to_tensor([16, 8, 2, 64]).view(1,-1))
    mypred3 = BP.pp(convert_x_list_to_tensor([16, 23, 19, 20]).view(1,-1))
    mypred4 = BP.pp(convert_x_list_to_tensor([16, 23, 19, 20, 16, 16]).view(1,-1))

    import bayes_num_game as bng
    H_all,log_prior_all,H_names = bng.generate_hypotheses(bng.mylambda_default)
    mypred1_bng = bng.bayesian_predictions(x_all, [16], H_all, log_prior_all)
    mypred2_bng = bng.bayesian_predictions(x_all, [16, 8, 2, 64], H_all, log_prior_all)
    mypred3_bng = bng.bayesian_predictions(x_all, [16, 23, 19, 20], H_all, log_prior_all)
    mypred4_bng = bng.bayesian_predictions(x_all, [16, 23, 19, 20, 16, 16], H_all, log_prior_all)

    assert(torch.allclose(mypred1,torch.tensor(mypred1_bng,dtype=mypred1.dtype,device=DEVICE)))
    assert(torch.allclose(mypred2,torch.tensor(mypred2_bng,dtype=mypred2.dtype,device=DEVICE)))
    assert(torch.allclose(mypred3,torch.tensor(mypred3_bng,dtype=mypred3.dtype,device=DEVICE)))
    assert(torch.allclose(mypred4,torch.tensor(mypred4_bng,dtype=mypred4.dtype,device=DEVICE)))
    print('**DEBUG** Passed debug with numpy number game code')

def train(mynet,fn_out_model,verbose=True,reg_strength = 1e-3,myopt='lbfgs'):
    mynet = mynet.to(device=DEVICE)
    if os.path.isfile(fn_out_model): 
         print('File exists: loading file',fn_out_model,'...')
         load_model(fn_out_model,mynet)
         print('number of parameters:',sum(p.numel() for p in mynet.parameters() if p.requires_grad))
         print('init loglike on val set (total):',-eval(mynet,val_data,val_data_eval,'none'))
         print('init loss on val set (mean):',eval(mynet,val_data,val_data_eval,'mean'))
         return
    if verbose:
        print('init loglike on val set (total):',-eval(mynet,val_data,val_data_eval,'none'))
        print('init loss on val set (mean):',eval(mynet,val_data,val_data_eval,'mean'))
    if myopt == 'lbfgs':
        fit_lbfgs(mynet,train_data,train_data_eval,reg_strength = reg_strength)
    elif myopt == 'adamw':
        fit_adamw(mynet,train_data,train_data_eval,reg_strength = reg_strength)
    elif myopt == 'dual_lbfgs':
        fit_dual_lbfgs(mynet,train_data,train_data_eval,reg_strength = reg_strength)
    else:
        assert False, "unknown optimizer "+myopt
    if verbose:
        print('number of parameters:',sum(p.numel() for p in mynet.parameters() if p.requires_grad))
        print('final loglike on val set (total):',-eval(mynet,val_data,val_data_eval,'none'))
        print('final loss on val set (mean):',eval(mynet,val_data,val_data_eval,'mean'))
        print('Some key parameters:')
        print('  lapse rate:',mynet.lapse_rate.item())
        print('  base rate:',mynet.b_rate.item())
        if hasattr(mynet,'log_prior'): print('  entropy of prior:',prior_entropy(mynet).detach().item())
        print('  reg strength:',mynet.reg_strength)
        if hasattr(mynet,'temp_ll'): print('  temperature of likelihood:',mynet.temp_ll.detach().item())
        if hasattr(mynet,'temp_lprior'): print('  temperature of prior:',mynet.temp_lprior.detach().item())
    save_model(fn_out_model,mynet)
    print('Saving:',fn_out_model)

def add_init_noise(net,sigma=0.01):
    # Add noise to the parameters of the model
    for p in net.parameters():
        p.data += sigma*torch.randn_like(p)


if __name__ == '__main__':

    debug()
    dir_model = 'out_models_alternative'
    if not os.path.exists(dir_model): os.makedirs(dir_model)
    train_data,train_data_eval = load_human_data('train')
    val_data,val_data_eval = load_human_data('val')    

    net_base = BayesNumGameNet()
    
    seed_all(0)
    net_tuned_prior = BayesNumGameOptPrior('bayes')
    train(net_tuned_prior, os.path.join(dir_model, 'bng_tuned_prior_sigma0.pt'))

    seed_all(0)
    net_tuned_full = BayesNumGameOptFull('bayes',scale_ll_init=3)
    train(net_tuned_full, os.path.join(dir_model, 'bng_tuned_full_sigma0.pt'))

    seed_all(0)
    net_tuned_full_temp = BayesNumGameOptFullTemp('bayes',scale_ll_init=3)
    train(net_tuned_full_temp, os.path.join(dir_model, 'bng_tuned_full_temp_sigma0.pt'), myopt='dual_lbfgs')    

    # # # Fit the prior only, with multiple reps and initial noise
    # # try_init_noise = [0.01,0.1,1.0]
    # # nrep = 5
    # # for r in range(1,nrep+1):
    # #     for sigma in try_init_noise:
    # #         seed_all(r)
    # #         net_tuned_prior = BayesNumGameOptPrior('bayes')
    # #         add_init_noise(net_tuned_prior,sigma)
    # #         train(net_tuned_prior, os.path.join(dir_model, 'bng_tuned_prior_sigma'+str(sigma)+'_run'+str(r)+'.pt'))
        
    # # # Fit the full model, with multiple reps and initial noise
    # # for r in range(1,nrep+1):
    # #     for sigma in try_init_noise:
    # #         seed_all(r)
    # #         net_tuned_full = BayesNumGameOptFull('bayes',scale_ll_init=3)
    # #         add_init_noise(net_tuned_full,sigma)
    # #         train(net_tuned_full, os.path.join(dir_model, 'bng_tuned_full_sigma'+str(sigma)+'_run'+str(r)+'.pt'))