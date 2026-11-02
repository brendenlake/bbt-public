import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import pickle
import sys
import argparse
import numpy as np
import math
import random
from copy import deepcopy
from model import BBT, describe_model
import datasets as dat
from train_utils import seed_all, extract, assert_consist_langs, display_input_output, get_best_device
from eval_utils import mean_masked_dim1, display_sys_gen_html_unmap, display_sys_gen_console_unmap
import matplotlib.pyplot as plt
from sklearn import decomposition, manifold
import pandas as pd
DEVICE = get_best_device()

## Evaluate a pre-trained model

def evaluate_ll(val_dataloader, net, langs, loss_fn=[], p_lapse=0.0, verbose=False, known_length=False):
    # Evaluate the total (sum) log-likelihood across the entire validation set
    # 
    # Input
    #   val_dataloader : 
    #   net : BBT model
    #   langs : dict of dat.Lang classes
    #   p_lapse : (default 0.) combine decoder outputs (prob 1-p_lapse) as mixture with uniform distribution (prob p_lapse)
    net.eval()
    total_N = 0
    total_ll = 0
    if not loss_fn: loss_fn = torch.nn.CrossEntropyLoss(ignore_index=langs['output'].PAD_idx)
    for batch_idx, val_batch in enumerate(val_dataloader):
        val_batch = dat.set_batch_to_device(val_batch)
        dict_loss = batch_ll(val_batch, net, loss_fn, langs, p_lapse=p_lapse, known_length=known_length)
        total_ll += dict_loss['ll']
        total_N += dict_loss['N']
    return total_ll, total_N


def evaluate_acc(val_dataloader, net, langs, max_length, eval_type='max', verbose=False):
    # Evaluate accuracy (exact match) across entire validation set
    #
    # Input
    #   val_dataloader : 
    #   net : BBT model
    #   langs : dict of dat.Lang classes
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; set of emission symbols we want to allow. Default of [] allows all output emissions
    net.eval()
    samples_pred = [] # list of all episodes with model predictions
    for batch_idx, val_batch in enumerate(val_dataloader): # each batch
        val_batch = dat.set_batch_to_device(val_batch)
        scores = batch_acc(val_batch, net, langs, max_length, eval_type=eval_type)
        samples_batch = val_batch['list_samples']
        for sidx in range(len(samples_batch)): # for each episode of the batch
            yq_sel = val_batch['q_idx'].cpu().numpy() == sidx # select for queries in this episode
            in_support = scores['in_support'][yq_sel] #numpy array
            is_novel = np.logical_not(in_support)
            v_acc = scores['v_acc'][yq_sel] #numpy array
            samples_batch[sidx]['yq_predict'] = extract(yq_sel, scores['yq_predict'])
            samples_batch[sidx]['v_acc'] = v_acc            
            samples_batch[sidx]['in_support'] = in_support #numpy array
            samples_batch[sidx]['acc_retrieve'] = np.mean(v_acc[in_support])*100.
            samples_batch[sidx]['acc_novel'] = np.mean(v_acc[is_novel])*100.
            samples_batch[sidx]['p_yes'] = scores['p_yes'][yq_sel]
        samples_pred.extend(samples_batch)

    # Compute mean accuracy across all val episodes
    mean_acc_retrieve = np.mean([sample['acc_retrieve'] for sample in samples_pred])
    v_acc_novel = np.array([sample['acc_novel'] for sample in samples_pred])
    mean_acc_novel = np.nanmean(v_acc_novel)
    if verbose:
        display_console_pred(samples_pred)
    return {'samples_pred':samples_pred, 'mean_acc_novel':mean_acc_novel, 'mean_acc_retrieve':mean_acc_retrieve, 'v_novel':v_acc_novel}

def evaluate_get_hooks(dataloader, net, langs, mytype):
    # Get internal summary representation of the model
    # 
    # Input
    #   dataloader : 
    #   net : BBT model.
    #   langs : dict of dat.Lang classes
    # Output
    #   hidden : [bsize, emb_size] tensor of hidden activations
    assert isinstance(net, BBT)
    assert mytype in ['set_concepts','number_game']
    assert(batch_size==1), "must have batch size 1"
    activation = {}
    from fit_bayes_number_game import BayesNumGameNet, convert_x_list_to_tensor
    BP_fast = BayesNumGameNet()
    BP_fast.to(device=DEVICE)
    def fast_entropy(x_as_list_of_strs): 
        # Helper function for computing entropy of a posterior distribution given x
        assert(all([isinstance(x,str) for x in x_as_list_of_strs]))
        x_as_list_of_ints = [int(item) for item in x_as_list_of_strs]
        data_t = convert_x_list_to_tensor(x_as_list_of_ints).view(1,-1)
        data_t = data_t.to(device=DEVICE)
        log_post = BP_fast.log_posterior(data_t)
        post = torch.exp(log_post).detach() # 1 x nhyp
        post = post.squeeze() # nhyp
        assert(post.numel()==len(BP_fast.H_all))
        post_nonzero = post[post>1e-4]
        entropy = -torch.sum(post_nonzero*torch.log(post_nonzero))
        return entropy.item()
    def get_activation_first_step(name):
        # Helper function for getting activation of an internal layer
        # Here, when used for the decoder, only get the first step's embedding (that is, assuming output is just 1 token)
        def hook(net, input, output):
            activation[name] = output.detach()[:,0]
        return hook
    def get_activation(name):
        # Helper function for getting activation of an internal layer
        def hook(net, input, output):
            activation[name] = output.detach()
        return hook
    def get_output_prob_yes_first_step(name):
        # Helper function for getting the probability of a 'yes' response across all queries
        def hook(net, input, output):
            first_output = output.detach()[:,0] # [nquery x output_size]
            activation[name] = F.softmax(first_output,dim=1)[:,langs['output'].symbol2index['yes']] # [nquery]
        return hook
    def get_output_prob_yes(name):
        # Helper function for getting the probability of a 'yes' response across all queries (not just the first step)
        def hook(net, input, output):
            all_output = output.detach() # [nquery x maxlen x output_size]
            activation[name] = F.softmax(all_output,dim=2)[:,:,langs['output'].symbol2index['yes']] # [nquery x maxlen x output_size]
        return hook
    def append_to_dict(mydict,mykey,myitem):
        if mykey not in mydict: mydict[mykey] = []
        mydict[mykey].append(myitem)

    if mytype=='set_concepts':        
        # for l in range(nlayers_encoder):
        #     net.transformer.encoder.layers[l].register_forward_hook(get_activation(f'encoder_layer{l}'))
        for l in range(nlayers_decoder):
            net.transformer.decoder.layers[l].register_forward_hook(get_activation_first_step(f'decoder_layer{l}'))
        net.out.register_forward_hook(get_output_prob_yes('prob_yes'))
    elif mytype=='number_game':
        for l in range(nlayers_decoder):
            net.transformer.decoder.layers[l].register_forward_hook(get_activation_first_step(f'decoder_layer{l}'))
        net.out.register_forward_hook(get_output_prob_yes_first_step('prob_yes'))
        for l in range(nlayers_encoder):
            net.transformer.encoder.layers[l].register_forward_hook(get_activation(f'encoder_layer{l}'))

    net.eval()
    dict_hooked = {}
    list_label_xs = []
    list_label_hyp = []
    list_label_ent = []
    list_xq = []
    for batch_idx, val_batch in enumerate(dataloader):
        if batch_idx % 20==0: print('batch',batch_idx,'of',len(dataloader))
        val_batch = dat.set_batch_to_device(val_batch)        
        _, memory_padding_mask, target_mask_no_eos = batch_get_internal(val_batch, net, langs) 
            # memory_padding_mask is [nquery x n_token] tensor, such that True means token is missing
            # target_mask is [nquery x max_length] tensor, such that True means token is missing
  
        list_xq.append(val_batch['xq'])
        x = activation['prob_yes'] # for numgame [nquery], or for set concepts, [nquery x maxlen]
        if mytype=='set_concepts':
            assert(x.dim() == 2)
            x = x[torch.logical_not(target_mask_no_eos)] # [nobj_flat]
            append_to_dict(dict_hooked,'prob_yes_flat',x)
        else:
            append_to_dict(dict_hooked,'prob_yes',x)
        
        for layer in range(nlayers_decoder):
            mytag = 'decoder_layer'+str(layer)
            x = activation[mytag] # [nquery x emb_size]
            x = torch.mean(x,dim=0) # [emb_size]
            append_to_dict(dict_hooked,mytag,x)

        # mean across tokens in time
        if mytype=='number_game':
            for layer in range(nlayers_encoder):
                mytag = 'encoder_layer'+str(layer)
                x = activation[mytag] # [nquery x n_token x emb_size]
                x = mean_masked_dim1(x,memory_padding_mask) # [nquery x emb_size]
                x = torch.mean(x,dim=0) # [emb_size]
                append_to_dict(dict_hooked,mytag+'_mean',x)

        # set up the labels
        samples_batch = val_batch['list_samples']
        mysample = samples_batch[0]
        if episode_type=='number_game_viz':
            assert(all([len(item)==1 for item in mysample['xs']])), 'check that each example is a single token'
            assert(all([mysample['xs']==s['xs'] for s in samples_batch])), 'check that all examples are the same'
            label_xs = ','.join([item[0] for item in mysample['xs']])
            list_label_ent.append(fast_entropy(label_xs.split(',')))
            list_label_xs.append(label_xs)
        label_hyp = str(mysample['aux']['h_str'])
        list_label_hyp.append(label_hyp)

    # stack the stored tensors
    for key in dict_hooked:
        if key != 'prob_yes_flat':
            dict_hooked[key] = torch.stack(dict_hooked[key])
    return dict_hooked, list_label_xs, list_label_hyp, list_label_ent, list_xq

def get_most_likely_hyp_via_sparse_approx_ngame(batch_prob_yes):
    # Get the most likely hypothesis for each posterior predictive
    # Input
    #  batch_prob_yes : [bsize, 101] tensor of probabilities of saying 'yes'
    # Output
    # list_best_hyp : list of strings, the most likely hypothesis for each posterior
    from sparse_pp_approx import fit_sparse_pp_approx_ngame
    from bayes_num_game import x_all
    bsize = batch_prob_yes.size(0)        
    x_eval = x_all[1:]
    print('Computing the sparse approximation to each posterior predictive')
    list_best_hyp = []
    for b in range(bsize):
        if b % 10 == 0: print('  approx',b,'of',bsize)
        pp_target = torch.stack((batch_prob_yes[b], 1-batch_prob_yes[b]),axis=1)
        try:
            dict_out = fit_sparse_pp_approx_ngame(pp_target, x_eval, verbose=False)
            list_best_hyp.append(dict_out['sorted_names'][0])
        except:
            list_best_hyp.append("nan")
    return list_best_hyp

def get_most_likely_hyp_via_sparse_approx_set_concepts(list_prob_yes_flat, list_xq, list_H, list_label_hyp_target=None, verbose=False):
    # Get the most likely hypothesis for each posterior predictive
    # Input
    #  list_prob_yes_flat : [bsize list], each is a tensor [nobj_flat] of probabilities of saying 'yes'
    #  list_xq : [bsize list], each is a nquery list of sets
    #  list_H : list of hypotheses (each a python function that takes in a list of sets and outputs a list of bools)
    #  list_label_hyp_target : [bsize list], each is a string, the target hypothesis for that example
    # Output
    # list_best_hyp : list of strings, the most likely hypothesis for each posterior
    from sparse_pp_approx import fit_sparse_pp_approx_set_concepts
    bsize = len(list_prob_yes_flat)
    print('Computing the sparse approximation to each posterior predictive')
    list_best_hyp = []
    for b in range(bsize):
        if b % 10 == 0: print('  approx',b,'of',bsize)
        pp_target = torch.stack((list_prob_yes_flat[b], 1-list_prob_yes_flat[b]), axis=1) # [nobj_flat, 2]
        try:
            dict_out = fit_sparse_pp_approx_set_concepts(pp_target, list_xq[b], list_H, verbose=verbose)
            label_hyp_infer = dict_out['sorted_names'][0]
            list_best_hyp.append(label_hyp_infer)
        except:
            list_best_hyp.append("nan")
    return list_best_hyp

def batch_ll(batch, net, loss_fn, langs, p_lapse=0.0, known_length=False):
    # Evaluate log-likelihood (average over cells, and sum total) for a given batch
    #
    # Input
    #   batch : from dat.make_bbt_batch
    #   loss_fn : loss function
    #   langs : dict of dat.Lang classes
    #   p_lapse : probability of making a random lapse response
    #   known_length : (true/false) should we ask the model to predict the EOS token?
    net.eval()
    m = len(batch['yq']) # b*nq
    target_batches = batch['yq_padded'] # b*nq x max_length
    target_shift = batch['yq_sos_padded'] # b*nq x max_length
        # Shifted targets with padding (added SOS symbol at beginning and removed EOS symbol) 
    if known_length: target_batches = replace_EOS_w_PAD(target_batches,langs['output'])

    decoder_output = net(target_shift, batch)
        # b*nq x max_length x output_size
    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (batch*max_len, output_size)
    if p_lapse > 0:
        if known_length:
            logits_flat = smooth_decoder_outputs(logits_flat,p_lapse,langs['output'].symbols,langs)
        else:
            logits_flat = smooth_decoder_outputs(logits_flat,p_lapse,langs['output'].symbols+[dat.EOS_token],langs)
    loss = loss_fn(logits_flat, target_batches.reshape(-1))
    loglike = -loss.cpu().item()
    # N = torch.sum(target_batches.reshape(-1) != langs['output'].PAD_idx, dim=None)
    N = torch.sum(target_batches.reshape(-1) != langs['output'].PAD_idx)
    
    dict_loss = {}
    dict_loss['ll_by_cell'] = loglike # average over cells    
    dict_loss['N'] = float(N) # total number of valid cells
    dict_loss['ll'] = dict_loss['ll_by_cell'] * dict_loss['N'] # total LL
    return dict_loss


def batch_get_internal(batch, net, langs):
    # Get internal representation at a certain layer
    #
    # Input
    #   batch : from dat.make_bbt_batch
    #   langs : dict of dat.Lang classes
    net.eval()
    m = len(batch['yq']) # b*nq
    target_batches = batch['yq_padded'] # b*nq x max_length
    target_shift = batch['yq_sos_padded'] # b*nq x max_length
        # Shifted targets with padding (added SOS symbol at beginning and removed EOS symbol) 
    target_mask_no_eos = (target_batches == langs['output'].PAD_idx) | (target_batches == langs['output'].EOS_idx)
        # b*nq x max_length, Mask such that false means leave alone
    memory, memory_padding_mask = net.encode(batch)
    # memory_padding_mask : [b*nq (batch_size) x maxlen_src] binary mask. False means to leave alone
    decoder_output = net.decode(target_shift, memory, memory_padding_mask)
        # b*nq x max_length x output_size
    myhidden = decoder_output[:,0]
    return myhidden, memory_padding_mask, target_mask_no_eos

def replace_EOS_w_PAD(x,mylang):
    # x is a tensor of token indices
    # replace each EOS with a PAD
    #
    # return
    #  x : [b*nq x max_length] tensor
    assert(x.dim()==2)    
    x[x==mylang.EOS_idx] = mylang.PAD_idx
    # x_length = torch.sum(x != mylang.PAD_idx, dim=1).tolist()
    return x

def smooth_decoder_outputs(logits_flat,p_lapse,lapse_symb_include,langs):
    # Mix decoder outputs (logits_flat) with uniform distribution over allowed emissions (in lapse_symb_include)
    #
    # Input
    #  logits_flat : (batch*max_len, output_size) # unnomralized log-probabilities
    #  p_lapse : probability of a uniform lapse
    #  lapse_symb_include : list of tokens (strings) that we want to include in the lapse model
    #  langs : dict of dat.Lang classes
    #
    # Output
    #  log_probs_flat : (batch*max_len, output_size) normalized log-probabilities
    lapse_idx_include = [langs['output'].symbol2index[s] for s in lapse_symb_include]
    assert dat.SOS_token not in lapse_symb_include # SOS should not be an allowed output through lapse model
    sz = logits_flat.size() # get size (batch*max_len, output_size)
    probs_flat = F.softmax(logits_flat,dim=1) # (batch*max_len, output_size)
    num_classes_lapse = len(lapse_idx_include)
    probs_lapse = torch.zeros(sz, dtype=torch.float)
    probs_lapse = probs_lapse.to(device=DEVICE)
    probs_lapse[:,lapse_idx_include] = 1./float(num_classes_lapse)
    log_probs_flat = torch.log((1-p_lapse)*probs_flat + p_lapse*probs_lapse) # (batch*max_len, output_size)
    return log_probs_flat

def batch_acc(batch, net, langs, max_length, eval_type='max', out_mask_allow=[]):
    # Evaluate exact match accuracy for a given batch
    #
    #  Input
    #   batch : from dat.make_bbt_batch
    #   net : BBT model
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; list of emission symbols (strings) we want to allow. Default of [] allows all output emissions
    assert eval_type in ['max','sample','sample_prob_yes']
    net.eval()
    emission_lang = langs['output']
    use_mask = len(out_mask_allow)>0
    memory, memory_padding_mask = net.encode(batch) 
        # memory : b*nq x maxlength_src x hidden_size
        # memory_padding_mask : b*nq x maxlength_src (False means leave alone)
    m = len(batch['yq']) # b*nq
    z_padded = torch.tensor([emission_lang.symbol2index[dat.SOS_token]]*m) # b*nq length tensor
    z_padded = z_padded.unsqueeze(1) # [b*nq x 1] tensor
    z_padded = z_padded.to(device=DEVICE)
    max_length_target = batch['yq_padded'].shape[1]-1 # length without EOS
    assert max_length >= max_length_target # make sure that the net can generate targets of the proper length

    # make the output mask if certain emissions are restricted
    if use_mask:
        assert dat.EOS_token in out_mask_allow # EOS must be included as an allowed symbol
        additive_out_mask = -torch.inf * torch.ones((m,net.output_size), dtype=torch.float)
        additive_out_mask = additive_out_mask.to(device=DEVICE)
        for s in out_mask_allow:
            sidx = langs['output'].symbol2index[s]
            additive_out_mask[:,sidx] = 0.

    # Run through decoder
    all_decoder_outputs = torch.zeros((m, max_length), dtype=torch.long)
    all_decoder_outputs = all_decoder_outputs.to(device=DEVICE)
    all_p_yes = torch.zeros((m, max_length), dtype=torch.float)
    for t in range(max_length):
        decoder_output = net.decode(z_padded, memory, memory_padding_mask)
            # decoder_output is b*nq x (t+1) x output_size
        decoder_output = decoder_output[:,-1] # get the last step's output (batch_size x output_size)
        if use_mask: decoder_output += additive_out_mask

        # Choose the symbols at next timestep
        if eval_type == 'max': # pick the most likely
            topi = torch.argmax(decoder_output,dim=1)
            emissions = topi.view(-1)
        elif eval_type == 'sample':
            emissions = Categorical(logits=decoder_output).sample()
        elif eval_type == 'sample_prob_yes':
            emissions = Categorical(logits=decoder_output).sample()
            norm_output = F.softmax(decoder_output,dim=1) # (batch_size x output_size)        
            norm_output = norm_output[:,emission_lang.symbol2index['yes']] # batch_size
            all_p_yes[:,t] = norm_output
        all_decoder_outputs[:,t] = emissions
        z_padded = torch.cat([z_padded, emissions.unsqueeze(1)], dim=1)

    # Get predictions as strings and see if they are correct
    all_decoder_outputs = all_decoder_outputs.detach()
    yq_predict = [] # list of all predicted query outputs as strings
    v_acc = np.zeros(m)
    for q in range(m):
        myseq = emission_lang.tensor_to_symbols(all_decoder_outputs[q,:].view(-1))
        yq_predict.append(myseq)
        v_acc[q] = yq_predict[q] == batch['yq'][q] # for each query, did model get it right?
    in_support = np.array(batch['in_support']) # which queries are also support items
    out = {'yq_predict':yq_predict, 'v_acc':v_acc, 'in_support':in_support, 'p_yes':all_p_yes.detach().numpy()}
    return out

def viz_train_dashboard(train_tracker):
    # Show loss curves
    import matplotlib.pyplot as plt
    if not train_tracker:
        print('No training stats to plot')
        return
    fv = lambda x : [t[x] for t in train_tracker]
    mystep = fv('step')
    mytrainloss = fv('avg_train_loss')
    myvalloss = fv('val_loss')
    print("Dashboard stats at step",mystep[-1])
    print(' final train loss:',round(fv('avg_train_loss')[-1],3))
    print(' final val loss:',round(fv('val_loss')[-1],3))
    print(' final learning rate:',fv('lr')[-1])
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(mystep,mytrainloss,'b',label='train')
    plt.plot(mystep,myvalloss,'r',label='val')
    plt.xlabel('step')
    plt.legend()
    plt.title('Loss')
    plt.subplot(2, 2, 2)
    plt.plot(mystep,fv('lr'),'b')
    plt.xlabel('step')
    plt.title('Learning rate')
    plt.show()

def evaluate_ll_multi_rep(nrep, val_dataloader, net, langs, p_lapse=0.0, known_length=False):
    # Across multiple random runs, evaluate the total (sum) log-likelihood across the validation set.
    #   Multiple runs may be needed when certain things are randomized, like the mapping between original data files
    #   and input/output tokens.
    #
    #  Input
    #   nrep : number of replications
    #    for rest of input arguments, see def evaluate_ll
    #   net : BBT model
    #   langs : dict of dat.Lang classes
    #
    # Output
    #   ave_ll_by_cell : return average log-likelihood for each cell
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=langs['output'].PAD_idx)
    list_ll = []
    list_N = []
    for i in range(nrep):
        if nrep >= 5 and (i % int(nrep/5)==0): print('  run',i)
        total_ll, total_N = evaluate_ll(val_dataloader, net, langs, loss_fn=loss_fn, p_lapse=p_lapse, known_length=known_length)        
        list_ll.append(total_ll)
        list_N.append(total_N)
    mean_ll = np.mean(list_ll)
    std_ll = np.std(list_ll)
    mean_N = np.mean(list_N)
    std_N = np.std(list_N)
    ave_ll_by_cell = np.sum(list_ll)/float(np.sum(list_N))
    print('  loglike: M =', round(mean_ll,4),'(SD=',round(std_ll,4),', Nrep=',nrep,') for', round(mean_N,4), '(SD=',round(std_N,4),') symbol predictions on average')
    print('    ave LL by cell: ', round(ave_ll_by_cell,4) )
    return {'ave_ll_by_cell': ave_ll_by_cell, 'ave_ll' : mean_ll, 'std_ll':std_ll, 'ave_N':mean_N}

def fit_p_lapse(nrep, val_dataloader, net, langs, greedy_stop=True, known_length=False):
    # Fit value for p_lapse. For each value, use evaluate_ll_multi_rep
    #  greedy_stop : stop when the objective no longer improves
    #
    # Output
    #  p_lapse : best fitting lapse value
    #  score_best : average log-like of best value
    #  mean_N : average number of evaluations 
    iter_p_lapse = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    iter_p_lapse += [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    score_best = {'ave_ll_by_cell':float(-math.inf)}
    for p_lapse in iter_p_lapse:
        seed_all()
        score_curr =  evaluate_ll_multi_rep(nrep, val_dataloader, net, langs, p_lapse=p_lapse, known_length=known_length)
        if score_curr['ave_ll_by_cell'] >= score_best['ave_ll_by_cell']:
            score_best = score_curr
            p_lapse_best = p_lapse
        elif greedy_stop:
            break # 
    return p_lapse_best, score_best

def fit_p_drop_example(nrep, net, langs, greedy_stop=False, known_length=False):
    # Fit value for p_drop_example. For each value, use evaluate_ll_multi_rep
    #  greedy_stop : stop when the objective no longer improves
    #
    # Output
    #  p_lapse : best fitting lapse value
    #  score_best : average log-like of best value
    #  mean_N : average number of evaluations 
    iter_p_drop = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    score_best = {'ave_ll_by_cell':float(-math.inf)}
    p_drop_orig =- D_val.p_drop_example
    for p_drop in iter_p_drop:
        print(' p_drop',p_drop,':')
        seed_all()
        D_val.set_p_drop(p_drop)
        val_dataloader_drop = DataLoader(D_val,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_bbt_batch(x,langs),shuffle=False)
        score_curr =  evaluate_ll_multi_rep(nrep, val_dataloader_drop, net, langs, p_lapse=ll_p_lapse, known_length=known_length)
        if score_curr['ave_ll_by_cell'] >= score_best['ave_ll_by_cell']:
            score_best = score_curr
            p_drop_best = p_drop
        elif greedy_stop:
            break #
    D_val.set_p_drop(p_drop_orig) # reset to original value
    return p_drop_best, score_best

def display_console_pred(samples_pred):
    # Print model predictions
    #
    # Input
    #  samples_pred : list of dicts from evaluate_acc, which has predicted query outputs for each episode
    for idx,sample in enumerate(samples_pred):
        print('Evaluation episode ' + str(idx))
        in_support = sample['in_support']
        is_novel = np.logical_not(in_support)
        print('  support items;')
        display_input_output(sample['xs'],sample['ys'],sample['ys'])
        print('  retrieval items;',round(sample['acc_retrieve'],3),'% correct')
        if sample['xq']:
            display_input_output(extract(in_support,sample['xq']),extract(in_support,sample['yq_predict']),extract(in_support,sample['yq']))
        else:
            display_input_output([],extract(in_support,sample['yq_predict']),extract(in_support,sample['yq']))
        print('  generalization items;',round(sample['acc_novel'],3),'% correct')
        if sample['xq']:
            display_input_output(extract(is_novel,sample['xq']),extract(is_novel,sample['yq_predict']),extract(is_novel,sample['yq']))
        else:
            display_input_output([],extract(is_novel,sample['yq_predict']),extract(is_novel,sample['yq'])) 
        if 'aux' in sample and 'h_str' in sample['aux']: print(sample['aux']['h_str'])


def display_p_yes(samples_pred, fs=20):
    # Make bar graph of BBT predictions vs. Bayesian model
    #
    # Input
    #  samples_pred : list of dicts from evaluate_acc, which has predicted query outputs for each episode
    import bayes_num_game as oracle
    for idx,sample in enumerate(samples_pred):
        H_all,log_prior_all = oracle.generate_hypotheses(oracle.mylambda_default)
        x_eval = np.array([int(x[0]) for x in sample['xq']])
        xs = [int(x[0]) for x in sample['xs']]
        mypred_bayes = oracle.bayesian_predictions(x_eval, xs, H_all, log_prior_all)
    
        fig, (ax1, ax2) = plt.subplots(2)
        # fig.set_figwidth(14)
        # fig.set_figheight(6.5)
        fig.set_figwidth(18)
        fig.set_figheight(5)
        oracle.subplot_predictions(ax1,x_eval,mypred_bayes)
        ax1.set_title('Bayesian inference, X='+str(xs))
        ax1.set_ylabel('prob. of membership')
        for item in ax1.get_xticklabels()+ax1.get_yticklabels(): item.set_fontsize(fs)

        mypred_bbt = sample['p_yes'][:,0] # we only care about first symbol
        oracle.subplot_predictions(ax2,x_eval,mypred_bbt)
        # ax2.set_title('neural net inference, X='+str(xs))
        ax2.set_ylabel('prob. of membership')
        for item in ax2.get_xticklabels()+ax2.get_yticklabels(): item.set_fontsize(fs)
        plt.savefig('fig_X='+str(xs)+'.png')
    # plt.show()

def write_set_acc_to_text(samples_pred):
    list_series = []
    for S in samples_pred: # for each episode (here, a set)
        yq_pred = S['yq_predict']
        yq = S['yq']
        p_yes = S['p_yes']
        fn = S['aux']['h_str'].split('\n')[-1]
        assert('fn=' in fn)
        fn = fn.lstrip('fn=').strip('"')
        fn = fn.split('/')[-1]
        fn = fn.strip('.txt')
        fn = fn.split('_')
        concept = fn[0]
        mylist = fn[1]
        set_number = str(int(fn[2])+1)
        n_sample = len(yq_pred)
        for s_idx in range(n_sample): # for each participant
            s_yq_pred = yq_pred[s_idx]
            s_yq = yq[s_idx]
            s_p_yes = p_yes[s_idx]
            for r_idx,mylabel in enumerate(s_yq): # for each response in a particular set
                myresponse = 'NaN'
                if r_idx < len(s_yq_pred): # response is available... that is, if the model didn't terminate early
                    myresponse = s_yq_pred[r_idx]
                    my_p_yes = s_p_yes[r_idx]
                series = pd.Series({'subject':'subj_'+str(s_idx),'response.number':r_idx+1,
                    'response':myresponse,'right.answer':mylabel,'p_yes':my_p_yes,
                    'concept':concept, 'list':mylist, 'set.number':set_number})
                list_series.append(series)
    df = pd.DataFrame(list_series)
    df['correct'] = df['response'] == df['right.answer']        
    fn_out_csv = 'ModelData-' + fn_out_model.lstrip(dir_model+'/').rstrip('.pt') + '-ACC.txt'
    df.to_csv(fn_out_csv, sep='\t', index=False)

def plot_input_embeddings(net,langs,type='PCA'):
    symbols = langs['input'].symbols
    v = torch.tensor([langs['input'].symbol2index[s] for s in symbols],dtype=torch.long)
    X = net.input_embedding(v).detach().numpy()
    if type=='PCA':
        pca = decomposition.PCA(n_components=2)
        X_new = pca.fit_transform(X)
    elif type=='tSNE':
        from openTSNE import TSNE
        tsne = TSNE(
            perplexity=30,
            metric="cosine",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        X_new = tsne.fit(X)
    else:
        assert False, "invalid embedding type"
    plt.clf()
    plt.scatter(X_new[:,0], X_new[:,1])
    for i, s in enumerate(symbols):
        plt.annotate(s, (X_new[i,0], X_new[i,1]))    
    plt.show()

def categorize_set_hyp(list_hyp_label):
    # Categorize hypotheses into different sets based on their labels
    #
    # get hypotheses that we want to highlight with different colors
    # Input
    # list_hyp_label : list of strings, each represents the underlying hypothesis class [N]
    def is_complex(h_str):
        quantifier_count = sum([h_str.count(q) for q in ['exists_(', 'forall_(', 'exists_exactly_one_(', 'exists_one_or_fewer_(']])
        has_connective = (' and_(' in h_str) or (' or_(' in h_str) or (' implies_(' in h_str) or (' iff_(' in h_str)
        return (quantifier_count >= 2) or (quantifier_count==1 and has_connective)
    def cbt(tag): return np.array([tag in h and not is_complex(h) for h in list_hyp_label]) # class by tag
    
    is_complex_hyp = {'label':'complex', 'sel':np.array([is_complex(h) for h in list_hyp_label]), 'color':'cyan'}
    is_and_hyp = {'label':'and', 'sel':cbt(' and_('), 'color':'magenta'}
    is_or_hyp = {'label':'or', 'sel':cbt(' or_('), 'color':'blue'}
    is_iff_hyp = {'label':'iff', 'sel':cbt(' iff_('), 'color':'green'}
    is_implies_hyp = {'label':'implies', 'sel':cbt(' implies_('), 'color':'purple'}
    is_forall_hyp = {'label':'forall', 'sel':cbt(' forall_('), 'color':'brown'}
    is_exists_hyp = {'label':'exists', 'sel':cbt(' exists_('), 'color':'orange'}
    is_exists1_hyp = {'label':'exists1', 'sel':cbt(' exists_one_or_fewer_('), 'color':'red'}
    # is_truefalse_hyp = {'label':'true/false', 'sel':cbt(' True') | cbt(' False'), 'color':'yellow'}
    is_true_hyp = {'label':'true', 'sel':cbt(' True'), 'color':'yellow'}
    is_false_hyp = {'label':'false', 'sel':cbt(' False'), 'color':'lime'}

    # Check that each set of hypotheses is mutually exclusive
    sets = [is_and_hyp, is_or_hyp, is_iff_hyp, is_implies_hyp, is_exists_hyp, is_forall_hyp, is_exists1_hyp, is_true_hyp, is_false_hyp, is_complex_hyp]
    sel_matrix = np.stack([s['sel'] for s in sets], axis=1)  # shape: [N, num_sets]
    
    # Verify each row has exactly one 1 (each hypothesis belongs to exactly one category)
    row_sums = np.sum(sel_matrix, axis=1)
    
    # Find hypotheses that appear in multiple categories
    multi_category_hyps = []
    for i, row_sum in enumerate(row_sums):
        if row_sum > 1:
            categories = [sets[j]['label'] for j in range(len(sets)) if sel_matrix[i, j]]
            multi_category_hyps.append((list_hyp_label[i], categories))
            print((list_hyp_label[i], categories))
    assert np.all(row_sums <= 1), f"Each hypothesis must belong to one or fewer categories. Violating rows are: {np.where(row_sums > 1)[0]}"            

    return sets, sel_matrix

def plot_hidden_embeddings_set_concepts(X,list_hyp_label,list_stat,type='PCA',alpha=0.5,min_std=0.05):
    # Input
    #  X : [N x emb_size] tensor
    #  list_hyp_label : list len N of strings, each represents the underlying hypothesis class
    #  list_stat :  (N x 2) array, where the first column is p(yes) and the second is entropy
    #  type : 'PCA' or 'tSNE' 
    #  alpha : transparency of points
    #  min_std : minimum standard deviation for PCA features (to avoid numerical instability)

    # get hypotheses that we want to highlight with different colors
    X = X.detach().cpu().numpy().astype(np.float32)
    sets, sel_matrix = categorize_set_hyp(list_hyp_label)
    x_std = X.std(axis=0)
    dim_to_keep = x_std >= min_std
    print('  ...excluding this many PCA dimensions with low std:', np.sum(~dim_to_keep))
    X_clean = X[:,dim_to_keep]
    print('  ...min std of remaining dimensions:', np.min(X_clean.std(axis=0)))
    pca_model = None
    if type=='PCA':
        pca = decomposition.PCA(n_components=2)
        try:
            pca_model = pca.fit(X_clean)
        except:
            print("Error occurred while fitting PCA")
            return None, None, None
        X_new = pca.transform(X_clean)
        print(' Key stats from PCA for:')
        print('  explained variance:',pca_model.explained_variance_ratio_)
        print('  explained variance total:',np.sum(pca_model.explained_variance_ratio_))

        print('Correlation with probability of yes answer:')
        print('  corr. PC0 and prob. yes',round(np.corrcoef(X_new[:,0],list_stat[:,0])[0,1],3))
        print('  corr. PC1 and prob. yes',round(np.corrcoef(X_new[:,1],list_stat[:,0])[0,1],3))
        print('Correlation with binary entropy of yes/no answer:')
        print('  corr. PC0 and ent. yes/no',round(np.corrcoef(X_new[:,0],list_stat[:,1])[0,1],3))
        print('  corr. PC1 and ent. yes/no',round(np.corrcoef(X_new[:,1],list_stat[:,1])[0,1],3))
        print("")

    elif type=='tSNE':
        from openTSNE import TSNE        
        tsne = TSNE(
            perplexity=30,
            metric="cosine",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        X_new = tsne.fit(X_clean)
    else:
        assert False, "invalid embedding type"
    fig, ax = plt.subplots(figsize=(4.5, 4.5))    
    ax.scatter(X_new[:,0], X_new[:,1], color='grey', alpha=0.1, label='feature')
    for hyp in sets:
        is_hyp = hyp['sel']
        ax.scatter(X_new[is_hyp, 0], X_new[is_hyp, 1], color=hyp['color'], alpha=alpha, label=hyp['label'])
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return pca_model, ax, dim_to_keep

def bernoulli_entropy(prob):
        prob = prob.float().clamp(min=1e-8, max=1.-1e-8)
        return -(prob*torch.log(prob) + (1.-prob)*torch.log(1.-prob))

def get_first_slot_pred_stats_set_concepts(list_prob_yes_flat, list_xq=None):
    # Return [mean p(yes), mean entropy] for query-object slot 1 in each episode.
    # With list_xq, both stats are computed across query sets within the episode.
    list_stat = []
    if list_xq is None:
        list_xq = [None]*len(list_prob_yes_flat)
    else:
        assert len(list_xq) == len(list_prob_yes_flat), "prob_yes_flat episodes and query episodes must align"
    for prob_yes_flat, xq in zip(list_prob_yes_flat, list_xq):
        if hasattr(prob_yes_flat, 'detach'):
            prob_yes_flat = prob_yes_flat.detach().cpu().view(-1)
        else:
            prob_yes_flat = torch.as_tensor(prob_yes_flat).view(-1)
        if prob_yes_flat.numel() < 1:
            list_stat.append([float('nan'), float('nan')])
            continue
        if xq is not None:
            query_lengths = [len(myxq) for myxq in xq]
            expected_len = sum(query_lengths)
            assert prob_yes_flat.numel() == expected_len, "prob_yes_flat and query objects must align"
            offset = 0
            first_slot = []
            for length in query_lengths:
                if length >= 1:
                    first_slot.append(prob_yes_flat[offset])
                offset += length
            if first_slot:
                first_slot = torch.stack(first_slot).float()
                first_slot_stat = [
                    first_slot.mean().item(),
                    bernoulli_entropy(first_slot).mean().item(),
                ]
            else:
                first_slot_stat = [float('nan'), float('nan')]
        else:
            first_slot_stat = [prob_yes_flat[0].item(), bernoulli_entropy(prob_yes_flat[0]).item()]
        list_stat.append(first_slot_stat)
    return np.array(list_stat, dtype=np.float32)

def plot_hidden_embeddings_ngame(X,list_hyp,list_xs,list_ent,list_prob_yes,type='PCA',alpha=0.5,min_std=0.05):
    # Input
    #  X : [N x emb_size] tensor
    #  list_hyp : list len N of strings, each represents the underlying hypothesis class
    #  list_xs : list len N of strings, each represents the input
    #  list_ent : list len N of floats, each represents the entropy of the posterior distribution over hypotheses
    #  list_prob_yes : list len N of floats, each represents the probability of the yes answer (averages across all queries of that concept)
    #  type : 'PCA' or 'tSNE'
    #  alpha : transparency of points
    #  min_std : minimum standard deviation for PCA features (to avoid numerical instability)
    import bayes_num_game as bng
    import ast
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    X = X.detach().cpu().numpy()
    is_nan_hyp = np.array([h == 'nan' for h in list_hyp])
    X = X[~is_nan_hyp]
    list_hyp = [h for j,h in enumerate(list_hyp) if not is_nan_hyp[j]]
    list_xs = [x for j,x in enumerate(list_xs) if not is_nan_hyp[j]]
    list_ent = np.array([e for j,e in enumerate(list_ent) if not is_nan_hyp[j]])
    list_prob_yes = np.array([p for j,p in enumerate(list_prob_yes) if not is_nan_hyp[j]])
    list_binary_ent_yes_no = -list_prob_yes*np.log(list_prob_yes)-(1-list_prob_yes)*np.log(1-list_prob_yes)
    print('...excluded this many points because of "NaN" hypotheses:',np.sum(is_nan_hyp))
    N = X.shape[0]
    list_xs_size = [len(x.split(',')) for x in list_xs]

    # get hypotheses that we want to highlight with different colors
    hyp_even_odd = ['evens','odds']
    hyp_powers_of_n = ['powers_of_'+str(j) for j in range(2,11)]
    hyp_multiples_of_n = ['mults_of_'+str(j) for j in range(3,13)]
    hyp_ending_in_n = ['ending_in_'+str(j) for j in range(0,10)]
    hyp_highlight = hyp_even_odd + hyp_powers_of_n + hyp_multiples_of_n + hyp_ending_in_n
    hyp_legend = ['evens','odds','powers_of_2','powers_of_10','mults_of_3','mults_of_6','mults_of_12','ending_in_1','ending_in_3','ending_in_6','ending_in_8']

    # get colormaps for different hypotheses
    color_even_odd = plt.get_cmap('Blues')(np.linspace(0.5, 1, len(hyp_even_odd)))
    color_powers_of_n = plt.get_cmap('Greens')(np.linspace(0.1, 1, len(hyp_powers_of_n)))
    color_multiples_of_n = plt.get_cmap('Oranges')(np.linspace(0.1, 1, len(hyp_multiples_of_n)))
    color_ending_in_n = plt.get_cmap('Purples')(np.linspace(0.1, 1, len(hyp_ending_in_n)))
    hyp_colors = np.concatenate([color_even_odd,color_powers_of_n,color_multiples_of_n,color_ending_in_n], axis=0)

    # process each datapoint according to which type of hypothesis is (based on hyp_highlight list, otherwise -1)
    myclass = []
    for i in range(N):
        myname = list_hyp[i]
        if myname in hyp_highlight:
            myclass.append(hyp_highlight.index(myname))
        else:
            # assert(',...,' in myname), print('invalid hypothesis name:',myname)
            myclass.append(-1)
    myclass = np.array(myclass)
    
    assert(X.shape[0] == len(list_hyp))
    x_std = X.std(axis=0)
    dim_to_keep = x_std >= min_std
    print('...excluding this many PCA dimensions with low std:', np.sum(~dim_to_keep))
    X_clean = X[:,dim_to_keep]
    if type=='PCA':   
        pca = decomposition.PCA(n_components=2)
        pca_model = pca.fit(X_clean)
        X_new = pca.transform(X_clean)
        print(' Key stats from PCA for:')
        print('  explained variance:',pca_model.explained_variance_ratio_)
        print('  explained variance total:',np.sum(pca_model.explained_variance_ratio_))

        print('Correlation with posterior entropy:')
        print('  corr. PC0 and post. entropy',round(np.corrcoef(X_new[:,0],list_ent)[0,1],3))
        print('  corr. PC1 and post. entropy',round(np.corrcoef(X_new[:,1],list_ent)[0,1],3))
        print('Correlation with probability of yes answer:')
        print('  corr. PC0 and prob. yes',round(np.corrcoef(X_new[:,0],list_prob_yes)[0,1],3))
        print('  corr. PC1 and prob. yes',round(np.corrcoef(X_new[:,1],list_prob_yes)[0,1],3))
        print('Correlation with binary entropy of yes/no answer:')
        print('  corr. PC0 and ent. yes/no',round(np.corrcoef(X_new[:,0],list_binary_ent_yes_no)[0,1],3))
        print('  corr. PC1 and ent. yes/no',round(np.corrcoef(X_new[:,1],list_binary_ent_yes_no)[0,1],3))
        print("")

        assert not np.isnan(X_new).any(), "NaN values found in PCA output"
    elif type=='tSNE':
        from openTSNE import TSNE
        tsne = TSNE(
            perplexity=30,
            metric="cosine",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        X_new = tsne.fit(X_clean)
    else:
        assert False, "invalid embedding type"

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    plt.clf()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    background = myclass == -1
    ax.scatter(X_new[background, 0], X_new[background, 1], color='yellow', alpha=alpha, label='interval')
    for c in range(len(hyp_highlight)):
        mysel = myclass == c
        if hyp_highlight[c] in hyp_legend:
            ax.scatter(X_new[mysel, 0], X_new[mysel, 1], color=hyp_colors[c], alpha=alpha,  label=hyp_highlight[c].replace('_',' '))
        else:
            ax.scatter(X_new[mysel, 0], X_new[mysel, 1], color=hyp_colors[c], alpha=alpha)
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    # # code for roll-over annotation of graph
    # annot = ax.annotate("", xy=(0,0), xytext=(20,20),
    #                     textcoords="offset points",
    #                     bbox=dict(boxstyle="round", fc="w"),
    #                     arrowprops=dict(arrowstyle="->"))
    # annot.set_visible(False)
    # def update_annot(ind):
    #     pos = scatter.get_offsets()[ind["ind"][0]]
    #     annot.xy = pos
    #     text = "{}".format("; ".join([np.array(list_xs)[background][n] for n in ind["ind"]]))
    #     annot.set_text(text)
    #     annot.get_bbox_patch().set_alpha(0.4)
    # def hover(event):
    #     vis = annot.get_visible()
    #     if event.inaxes == ax:
    #         cont, ind = scatter.contains(event)
    #         if cont:
    #             update_annot(ind)
    #             annot.set_visible(True)
    #             fig.canvas.draw_idle()
    #         else:
    #             if vis:
    #                 annot.set_visible(False)
    #                 fig.canvas.draw_idle()
    # fig.canvas.mpl_connect("motion_notify_event", hover)

    # ax.legend(framealpha=1.)
    # plt.show()
    return pca_model, ax, dim_to_keep



def make_sparse_approx(samples_pred, inc_threshold=1e-2, make_plots=False):
    # Make bar graph of Bayesian model vs BBT model, 
    #  such that hypotheses for BBT are inferred using sparse approximation
    #
    # Input
    #  samples_pred : list of dicts from evaluate_acc, which has predicted query outputs for each episode
    #  inc_threshold : threshold for displaying a hypothesis after the sparse approximation
    #  make_plots : if True, make plots for each episode
    # Return
    #  list_hyp_diff : list of list of dicts, each with the difference in posterior probabilities for each hypothesis
    import bayes_num_game as oracle
    from sparse_pp_approx import fit_sparse_pp_approx_ngame
    list_hyp_diff = []
    for idx,sample in enumerate(samples_pred):

        # get bayesian predictions
        H_all,log_prior_all,H_names = oracle.generate_hypotheses(oracle.mylambda_default)
        x_eval = np.array([int(x[0]) for x in sample['xq']])
        xs = [int(x[0]) for x in sample['xs']]
        post_bayes = np.exp(oracle.log_posterior(xs, H_all, log_prior_all))
        post_bayes = torch.tensor(post_bayes,dtype=torch.float)
        pp_bayes_numpy = oracle.bayesian_predictions(x_eval, xs, H_all, log_prior_all)

        # get BBT predictions
        mypred_bbt = torch.tensor(sample['p_yes'][:,0], device=DEVICE) # we only care about first symbol
        pp_bbt = torch.stack((mypred_bbt,1-mypred_bbt),dim=1)
        struct_w = fit_sparse_pp_approx_ngame(pp_bbt,torch.tensor(x_eval,dtype=torch.long), verbose=False)
        pp_sparse = struct_w['pp_approx']
        sorted_post_sparse = struct_w['sorted_w']
        sorted_names_sparse = struct_w['sorted_names']
        to_inc = sorted_post_sparse >= inc_threshold
        sorted_post_sparse = sorted_post_sparse[to_inc]
        sorted_names_sparse = [sorted_names_sparse[i] for i in range(len(sorted_names_sparse)) if to_inc[i]]

        # compare with bayes and compute differences
        idx_names_orig = [H_names.index(name) for name in sorted_names_sparse]
        sorted_post_bayes = post_bayes[idx_names_orig]
        sorted_post_diff = torch.abs(sorted_post_sparse - sorted_post_bayes)

        # sort by posterior probabilities for different hypotheses
        # idx_re_sort = torch.argsort(sorted_post_diff,descending=True)
        # sorted_post_sparse = sorted_post_sparse[idx_re_sort]
        # sorted_post_bayes = sorted_post_bayes[idx_re_sort]  
        # sorted_post_diff =  sorted_post_diff[idx_re_sort]
        # sorted_post_names = [sorted_names_sparse[i] for i in idx_re_sort]

        list_s = []
        for j in range(len(sorted_names_sparse)): # for each hypothesis
            s = {}
            my_h = torch.nonzero(torch.tensor(H_all[H_names.index(sorted_names_sparse[j])]))
            s['h_nonzero'] = my_h
            s['h_name'] = sorted_names_sparse[j]
            s['diff'] = sorted_post_diff[j]
            s['post_bayes'] = sorted_post_bayes[j].item()
            s['post_bbt_sparse'] = sorted_post_sparse[j].item()
            list_s.append(s)

            if make_plots:
                print("")
                print('Hyp=',s['h_name'])
                print('  BBT:',round(s['post_bbt_sparse'],3))
                print('  Bayes:',round(s['post_bayes'],3))

                fig, (ax1, ax2, ax3) = plt.subplots(3)
                fig.set_figwidth(18)
                fig.set_figheight(5)
                oracle.subplot_predictions(ax1,x_eval,pp_bayes_numpy,highlight=my_h)
                ax1.set_title('Bayesian inference, X='+str(xs),'Hyp='+s['h_name'])
                ax1.set_ylabel('prob. of membership')

                oracle.subplot_predictions(ax2,x_eval,pp_bbt[:,0],highlight=my_h)            
                oracle.subplot_predictions(ax3,x_eval,pp_sparse[:,0],highlight=my_h)
                plt.show()
    list_hyp_diff.append(list_s)
    return list_hyp_diff

        
if __name__ == "__main__":
        # Adjustable parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--fn_out_model', type=str, default='', help='*REQUIRED*. Filename for loading the model')
        parser.add_argument('--dir_model', type=str, default='out_models', help='Directory for loading the model file')
        parser.add_argument('--max_length_eval', type=int, default=10, help='Maximum generated sequence length')
        parser.add_argument('--batch_size', type=int, default=-1, help='Number of episodes in batch')                                
        parser.add_argument('--episode_type', type=str, default='', help='What type of episodes do we want? See datasets.py for options')
        parser.add_argument('--dashboard', default=False, action='store_true', help='Showing loss curves during training.')
        parser.add_argument('--ll', default=False, action='store_true', help='Evaluate log-likelihood of validation (val) set')
        parser.add_argument('--ll_oracle', default=False, action='store_true', help='Evaluate log-likelihood of validation (val) set using ORACLE model')
        parser.add_argument('--compare_special', default=False, action='store_true', help='Evaluate log-likelihood of validation (val) set using ORACLE model')
        parser.add_argument('--get_internal_num_game', default=False, action='store_true', help='get internal representations of the model')
        parser.add_argument('--get_internal_set_concepts', default=False, action='store_true', help='get internal representations of the model')
        parser.add_argument('--infer_labels_ftune', default=False, action='store_true', help='for --get_internal, label the hypotheses according to the human tuned model')
        parser.add_argument('--max', default=False, action='store_true', help='Find best outputs for val commands (greedy decoding)')
        parser.add_argument('--sample', default=False, action='store_true', help='Sample outputs for val commands')
        parser.add_argument('--set_acc_to_text', default=False, action='store_true', help='Sample outputs and send to dataframe')
        parser.add_argument('--sys_gen_html', default=False, action='store_true', help='Generate html file for systmatic gen task')
        parser.add_argument('--fit_lapse', default=False, action='store_true', help='Fit the best lapse rate according to log-likelihood on validation')
        parser.add_argument('--fit_p_drop_example', default=False, action='store_true', help='Fit the best drop example rate according to log-likelihood on validation')
        parser.add_argument('--ll_nrep', type=int, default=1, help='Evaluate each episode this many times when computing log-likelihood (needed for stochastic remappings)')
        parser.add_argument('--ll_p_lapse', type=float, default=0., help='Lapse rate when evaluating log-likelihoods')
        parser.add_argument('--known_length', default=False, action='store_true', help='for log-like evaluations, do not predict EOS')
        parser.add_argument('--verbose', default=False, action='store_true', help='Inspect outputs in more detail')

        args = parser.parse_args()
        fn_out_model = args.fn_out_model
        dir_model = args.dir_model
        max_length_eval = args.max_length_eval
        episode_type = args.episode_type
        do_dashboard = args.dashboard
        batch_size = args.batch_size
        do_ll = args.ll
        do_ll_oracle = args.ll_oracle
        do_get_internal_num_game = args.get_internal_num_game
        do_get_internal_set_concepts = args.get_internal_set_concepts
        do_infer_labels_ftune = args.infer_labels_ftune
        do_compare = args.compare_special
        do_max_acc = args.max
        do_sample_acc = args.sample
        do_fit_lapse = args.fit_lapse
        do_fit_p_drop_example = args.fit_p_drop_example
        do_known_length = args.known_length
        do_set_acc_to_txt = args.set_acc_to_text
        do_pca_embeddings = False
        do_sys_gen_sample_html = args.sys_gen_html
        ll_nrep = args.ll_nrep
        ll_p_lapse = args.ll_p_lapse
        verbose = args.verbose

        model_tag = episode_type + '_' + fn_out_model.replace('.pt','')
        fn_out_model = os.path.join(dir_model, fn_out_model)
        if not os.path.isfile(fn_out_model):
             raise Exception('filename '+fn_out_model+' not found')

        seed_all()
        print('Loading model:',fn_out_model,'on',DEVICE)
        checkpoint = torch.load(fn_out_model, map_location=DEVICE, weights_only=False)
        if not episode_type: episode_type = checkpoint['episode_type']
        if batch_size<=0: batch_size = checkpoint['batch_size']
        nets_state_dict = checkpoint['nets_state_dict']
        if list(nets_state_dict.keys())==['net']: nets_state_dict = nets_state_dict['net'] # for compatibility with legacy code
        input_size = checkpoint['langs']['input'].n_symbols
        output_size = checkpoint['langs']['output'].n_symbols
        emb_size = checkpoint['emb_size']
        if 'max_ns' in checkpoint: max_ns = checkpoint['max_ns']
        dropout_p = checkpoint['dropout']
        ff_mult = checkpoint['ff_mult']
        myact = checkpoint['activation']
        nlayers_encoder = checkpoint['nlayers_encoder']
        nlayers_decoder = checkpoint['nlayers_decoder']
        train_tracker = checkpoint['train_tracker']
        best_val_loss = -float('inf')
        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']        

        print(' Loading model that has completed (or started) ' + str(checkpoint['epoch']) + ' of ' + str(checkpoint['nepochs']) + ' epochs')
        print('  test episode_type:',episode_type)
        print('  batch size:',checkpoint['batch_size'])
        print('  patience:',checkpoint['args'].patience)
        print('  max eval length:', max_length_eval)
        print('  number of steps:', checkpoint['step'])
        print('  best val loss achieved: {:.4f}'.format(best_val_loss))
        print('  evaluation with batch size:',batch_size)
        if 'grad_clip' in checkpoint['args']:
            print('  grad_clip:',checkpoint['args'].grad_clip)

        # Load validation dataset
        D_train,D_val = dat.get_dataset(episode_type)
        langs = D_val.langs
        assert(D_val.max_ns <= max_ns), "evaluating model on more support examples than during training" 
        assert_consist_langs(langs,checkpoint['langs'])
        train_dataloader = DataLoader(D_train,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_bbt_batch(x,langs),shuffle=False)
        val_dataloader = DataLoader(D_val,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_bbt_batch(x,langs),shuffle=False)
        
        
        # Load model parameters         
        net = BBT(emb_size, input_size, output_size,
            langs['input'].PAD_idx, langs['output'].PAD_idx,
            nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
            dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)
        net.load_state_dict(nets_state_dict)
        describe_model(net)
        net = net.to(device=DEVICE)

        # Perform selected evaluations
        if do_dashboard:
            print('Showing loss curves during training <close plot to continue>')
            viz_train_dashboard(train_tracker)
        if do_ll and ll_nrep==1:
            import time
            t0 = time.time()
            seed_all()
            print('Evaluating log-likelihood of val episodes...')
            print('  with lapse rate',ll_p_lapse)
            if do_known_length: print('  with known length')
            total_ll,total_N = evaluate_ll(val_dataloader, net, langs, p_lapse=ll_p_lapse, known_length=do_known_length)
            print('evaluation on',episode_type,'loglike:',round(total_ll,4),'for',int(total_N),'symbol predictions')
            print('mean loglike is',round(total_ll/total_N,5),'per symbol')
            t1 = time.time()
            print('  (time taken:',round(t1-t0,2),'seconds)')
        if do_ll_oracle:
            seed_all()
            print('ORACLE evaluation of log-likelihood of val episodes...')
            print('  with lapse rate',ll_p_lapse)
            from bayes_number_game_torch import BayesNumGameNet
            BP = BayesNumGameNet(output_size=output_size, langs=langs)
            total_ll_torch,total_N_torch = evaluate_ll(val_dataloader, BP, langs, p_lapse=ll_p_lapse, known_length=True)
            print('torch* evaluation on',episode_type,'loglike:',round(total_ll_torch,4),'for',int(total_N_torch),'symbol predictions')
            print('torch* mean loglike is',round(total_ll_torch/total_N_torch,5),'per symbol')
        if do_ll and ll_nrep>1:
            seed_all()
            print('Evaluating log-likelihood of val episodes...')
            print('  with lapse rate',ll_p_lapse)
            print('  replicated across',ll_nrep,'random runs/permutations')
            if do_known_length: print('  with known length')
            evaluate_ll_multi_rep(ll_nrep, val_dataloader, net, langs, p_lapse=ll_p_lapse, known_length=do_known_length)
        if do_compare: ## compare number game model to oracle
            assert(episode_type=='number_game_special')
            seed_all()
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='sample_prob_yes')
            display_p_yes(E['samples_pred'])
        if do_max_acc:
            seed_all()
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='max', verbose=verbose)
            print('Evaluating set of validation episodes (via greedy decoding)...')
            print(' Acc Retrieve (val):',round(E['mean_acc_retrieve'],4))
            v_novel = E['v_novel']
            if np.any(np.isnan(v_novel)):
                print("Warning: a total of", np.sum(np.isnan(v_novel)), "NaN values found in novel accuracy. These will be excluded from the mean and SD calculations.")
                v_novel = v_novel[~np.isnan(v_novel)]
            print(' Acc Novel (val):',round(np.mean(v_novel),4),
                    'SD=',round(np.std(v_novel),4),'N=',len(v_novel))
        if do_sample_acc:
            seed_all()
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='sample', verbose=verbose)
            print('Evaluating set of validation episodes (via sampling)...')
            print(' Acc Retrieve (val):',round(E['mean_acc_retrieve'],4))
            print(' Acc Novel (val):',round(np.mean(E['v_novel']),4),
                    'SD=',round(np.std(E['v_novel']),4),'N=',len(E['v_novel']))
        if do_fit_lapse:
            print('Fitting for the best value of p_lapse use log-like...')
            if do_known_length: print('  with known length')
            print('  Each value is replicated across',ll_nrep,'random runs/permutations')
            # assert(ll_nrep>1), "should use more than one replication for parameter fitting"
            p_lapse_best, score_best = fit_p_lapse(ll_nrep, val_dataloader, net, langs, known_length=do_known_length)
            print('* BEST FIT * p_lapse=',p_lapse_best,'with mean loglike score of',round(score_best['ave_ll'],4),
                  '(or',round(score_best['ave_ll_by_cell'],4),'per cell)')
        if do_fit_p_drop_example:
            print('Fitting for the best value of *exemplar drop* using log-like...')
            if do_known_length: print('  with known length')
            print('  Each value is replicated across',ll_nrep,'random runs/permutations')
            assert(ll_nrep>1), "should use more than one replication for parameter fitting"
            p_drop_example, score_best = fit_p_drop_example(ll_nrep, net, langs, known_length=do_known_length)
            print('* BEST FIT * p_drop_example=',p_drop_example,'with mean loglike score of',round(score_best['ave_ll'],4),
                  '(or',round(score_best['ave_ll_by_cell'],4),'per cell)')
        if do_set_acc_to_txt: # only for Piantadosi set concepts
            assert(episode_type=='set_concepts_gold_test')
            seed_all()
            data_multiplyer = 5 # number of times to multiply the dataset
            samples_pred = []
            for d_idx in range(1,data_multiplyer+1):
                print('pass through dataset:',d_idx,'of',data_multiplyer)
                E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='sample_prob_yes', verbose=verbose)
                samples_pred += E['samples_pred']
            write_set_acc_to_text(samples_pred)
        if do_pca_embeddings:
            plot_input_embeddings(net,langs)
        if do_get_internal_set_concepts:
            emb_type = 'PCA' # 'PCA' # or 'tSNE'
            assert(episode_type=='set_concepts_viz')
            fn_out_pkl = 'ModelInternal-' + fn_out_model.replace(dir_model+'/','').replace('.pt','.pkl')
            fn_use_label_infer = ''            
            if do_infer_labels_ftune:
                fn_use_label_infer = fn_out_pkl.replace('base','bbt').replace('paired','top')
            do_pkl = True
            if os.path.isfile(fn_out_pkl):
                response = input(f"File {fn_out_pkl} already exists. Do you want to overwrite it? (yes/no): ")
                if response.lower() == 'yes':
                    pass
                else:
                    do_pkl = False
                    print('Loading file...',fn_out_pkl)
                    dict_out = torch.load(fn_out_pkl)
                    dict_hooked = dict_out['dict_hooked']
                    list_label_hyp = dict_out['list_label_hyp']
                    list_label_infer_hyp = dict_out['list_label_infer_hyp']
                    list_xq = dict_out['list_xq']
            if do_pkl:
                print("Getting model internals...")
                dict_hooked, _, list_label_hyp, _, list_xq = evaluate_get_hooks(val_dataloader, net, langs, mytype='set_concepts')
                with open('data/set_concepts_viz/hyp_total_dict.pkl', 'rb') as f:
                    list_H = pickle.load(f)
                    list_H = [list_H[k] for k in sorted(list_H.keys())]
                list_label_hyp = [s.split('\n')[0].lstrip('h_str=').strip('"') for s in list_label_hyp]
                list_label_infer_hyp = get_most_likely_hyp_via_sparse_approx_set_concepts(dict_hooked['prob_yes_flat'], list_xq, list_H, list_label_hyp_target=list_label_hyp)
                dict_out = {'dict_hooked':dict_hooked, 'list_label_hyp':list_label_hyp, 'list_label_infer_hyp':list_label_infer_hyp, 'list_xq':list_xq}
                print('Writing output to...',fn_out_pkl)
                torch.save(dict_out, fn_out_pkl)
            if fn_use_label_infer:
                print('Using inferred labels from',fn_use_label_infer)
                dict_out = torch.load(fn_use_label_infer)
                assert(dict_out['list_label_hyp'] == list_label_hyp), "the original labels should be the same for the ftuned model and the base model, otherwise the inferred labels won't be comparable"
                list_label_infer_hyp = dict_out['list_label_infer_hyp']
            else:
                matches = [h == inf for h, inf in zip(list_label_hyp, list_label_infer_hyp)]
                percent_match = 100.0 * sum(matches) / len(matches) if matches else 0.0
                print(f"  percentage of matches between list_label_hyp and list_label_infer_hyp: {percent_match:.2f} (n={len(matches)})")
            fig_dir = 'figs_embedding'
            if not os.path.exists(fig_dir): os.makedirs(fig_dir)
            list_prob_yes_flat = dict_hooked['prob_yes_flat'] # list of size [batch], each is a flat vector for all set judgments in the query set
            list_stat = get_first_slot_pred_stats_set_concepts(list_prob_yes_flat, list_xq) # list of size [N x 2], where the first column is p(yes) and second is entopry
            
            # Add these special highlighted hypotheses
            _,D_special = dat.get_dataset('set_concepts_viz_marker')
            special_dataloader = DataLoader(D_special,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_bbt_batch(x,langs),shuffle=False)
            dict_hooked_special, _, list_label_hyp, _, _ = evaluate_get_hooks(special_dataloader, net, langs, mytype='set_concepts')            
            
            for name in dict_hooked.keys():
                if name == 'prob_yes_flat': continue # skip this one
                if 'encoder' in name: continue # skip this one
                fig_name = f"{fig_dir}/fig_embeddings_{model_tag}_{name}_{emb_type}_infer.pdf"
                if do_infer_labels_ftune: fig_name = fig_name.replace('base','ftune')
                print('\nPlotting',fig_name)
                pca_model, ax, dim_to_keep = plot_hidden_embeddings_set_concepts(dict_hooked[name],list_label_infer_hyp,list_stat,type=emb_type)
                if pca_model:
                    x_input_special = dict_hooked_special[name].cpu().numpy()
                    x_input_special = x_input_special[:,dim_to_keep]
                    X_special = pca_model.transform(x_input_special)
                    ax.scatter(X_special[:, 0], X_special[:, 1], color='k')
                    offset = 1.0
                    # for i, s in enumerate(list_label_hyp):
                    #     m = __import__('re').search(r'(hg\d+)_L2', s)
                    #     if m: s = m.group(1)
                    #     ax.annotate(s, (X_special[i, 0], X_special[i, 1]+offset), bbox=dict(facecolor='white'))
                    plt.savefig(fig_name)
                # plt.savefig(f"{fig_dir}/fig_{model_tag}_pie_infer.pdf")
                ## original labels
                # fig_name2 = f"{fig_dir}/fig_embeddings_{model_tag}_{name}_{emb_type}_original.pdf"
                # plot_hidden_embeddings_set_concepts(dict_hooked[name],list_label_hyp,list_prob_yes_flat,type=emb_type)
                # plt.savefig(fig_name2)
                # plt.savefig(f"{fig_dir}/fig_{model_tag}_pie_original.pdf")
        if do_get_internal_num_game:
            import ast
            import bayes_num_game as bng
            emb_type = 'PCA' # or 'tSNE'
            fn_out_pkl = 'ModelInternal-' + fn_out_model.replace(dir_model+'/','').replace('.pt','.pkl')
            fn_use_label_infer = ''          
            if do_infer_labels_ftune:
                fn_use_label_infer = fn_out_pkl.replace('base','bbt').replace('paired','top')
            print(episode_type)
            assert(episode_type=='number_game_viz')            
            do_pkl = True
            if os.path.isfile(fn_out_pkl):
                response = input(f"File {fn_out_pkl} already exists. Do you want to overwrite it? (yes/no): ")
                if response.lower() == 'yes':
                    pass
                else:
                    do_pkl = False
                    print('Loading file...',fn_out_pkl)
                    dict_out = torch.load(fn_out_pkl)
                    dict_hooked = dict_out['dict_hooked']
                    list_label_xs = dict_out['list_label_xs']
                    list_label_hyp = dict_out['list_label_hyp']    
                    list_label_infer_hyp = dict_out['list_label_infer_hyp']
                    list_label_ent = dict_out['list_label_ent']
            if do_pkl:
                print("Getting model internals...")
                dict_hooked, list_label_xs, list_label_hyp, list_label_ent, list_xq = evaluate_get_hooks(val_dataloader, net, langs, mytype='number_game')
                list_label_infer_hyp = get_most_likely_hyp_via_sparse_approx_ngame(dict_hooked['prob_yes'])
                dict_out = {'dict_hooked':dict_hooked, 'list_label_xs':list_label_xs,
                            'list_label_hyp':list_label_hyp, 'list_label_infer_hyp':list_label_infer_hyp, 'list_label_ent':list_label_ent}
                print('Writing output to...',fn_out_pkl)
                torch.save(dict_out, fn_out_pkl)
            list_label_hyp = [bng.list_of_num_to_hname(ast.literal_eval(h)) for h in list_label_hyp] # convert to name of hypotheses name
            if fn_use_label_infer:
                print('Using inferred labels from',fn_use_label_infer)
                dict_out = torch.load(fn_use_label_infer)
                list_label_infer_hyp = dict_out['list_label_infer_hyp']
            else:
                matches_rules = [h == inf for h, inf in zip(list_label_hyp, list_label_infer_hyp) if h[0] != '[']
                percent_match_rules = 100.0 * sum(matches_rules) / len(matches_rules) if matches_rules else 0.0
                print(f" math hypotheses: percentage of matches between list_label_hyp and list_label_infer_hyp: {percent_match_rules:.2f} (n={len(matches_rules)})%")            
                matches_interval = [h == inf for h, inf in zip(list_label_hyp, list_label_infer_hyp) if h[0] == '[']
                percent_match_interval = 100.0 * sum(matches_interval) / len(matches_interval) if matches_interval else 0.0
                print(f" interval hypotheses: percentage of matches between list_label_hyp and list_label_infer_hyp: {percent_match_interval:.2f} (n={len(matches_interval)})%")            
            
            # Add these special highlighted hypotheses
            _,D_special = dat.get_dataset('number_game_special')
            special_dataloader = DataLoader(D_special,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_bbt_batch(x,langs),shuffle=False)
            dict_hooked_special, list_label_xs_special, _, _, _ = evaluate_get_hooks(special_dataloader, net, langs, mytype='number_game')
            fig_dir = 'figs_embedding'
            if not os.path.exists(fig_dir): os.makedirs(fig_dir)
            for name in dict_hooked_special.keys():
                if 'decoder' not in name: continue
                fig_name = f"{fig_dir}/fig_{model_tag}_embeddings_{name}_{emb_type}.pdf"
                if do_infer_labels_ftune: fig_name = fig_name.replace('base','bbt')
                print('\nplotting',fig_name)
                list_prob_yes = dict_hooked['prob_yes'].cpu().mean(dim=1).numpy() # [batch, nquery] -> [batch]
                pca_model, ax, dim_to_keep = plot_hidden_embeddings_ngame(dict_hooked[name],list_label_infer_hyp,list_label_xs,list_label_ent,list_prob_yes,type=emb_type)
                x_input_special = dict_hooked_special[name].cpu().numpy()
                x_input_special = x_input_special[:,dim_to_keep]
                X_special = pca_model.transform(x_input_special)
                ax.scatter(X_special[:, 0], X_special[:, 1], color='k')
                offset = 1.0
                for i, s in enumerate(list_label_xs_special):
                    s = r'$\{' + s + r'\}$'
                    ax.annotate(s, (X_special[i, 0], X_special[i, 1]+offset), bbox=dict(facecolor='white'))
                plt.savefig(fig_name)
        if do_sys_gen_sample_html:
            seed_all()
            print('Sampling from model to produce HTML file...')
            assert(episode_type == 'sys_gen_test_human_mult10')
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='sample', verbose=False)            
            episode_type_tag = episode_type
            os.makedirs('analysis_sys_gen/'+episode_type_tag, exist_ok=True)            
            with open('analysis_sys_gen/'+episode_type_tag+'/'+model_tag+'.txt','w') as fid_out:
                prev = sys.stdout
                sys.stdout = fid_out # re-rout outputs to file
                display_sys_gen_console_unmap(E['samples_pred'])
                sys.stdout = prev
            with open('analysis_sys_gen/'+episode_type_tag+'/template.html','r') as fid_in:
                mylines = fid_in.readlines()
            with open('analysis_sys_gen/'+episode_type_tag+'/'+model_tag+'.html','w') as fid_out:
                for l in mylines:
                    fid_out.write(l)
                    if l.strip() == '// PLACEHOLDER':
                        fid_out.write('var title="'+model_tag+'"; \n')
                        if 'probe' in episode_type_tag:
                            display_sys_gen_html_unmap(E['samples_pred'], fid_out, freq='percent', include_support=True)
                        else:    
                            display_sys_gen_html_unmap(E['samples_pred'], fid_out, freq='percent')
            print('Done writing to files: analysis_sys_gen/'+model_tag+'.html and .txt')
