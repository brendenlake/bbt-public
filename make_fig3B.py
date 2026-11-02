import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import os
import sys
import argparse
import numpy as np
import math
from copy import deepcopy
from model import BBT, describe_model
import datasets as dat
from train_utils import seed_all, extract, assert_consist_langs, display_input_output, get_best_device
import matplotlib.pyplot as plt
import glob
import bayes_num_game as oracle
import eval as meval
from scipy.stats import pearsonr
DEVICE = get_best_device()

# For each episode of human data, compare BBT with the alternative models.
#  if do_sparse_decomp, we do a decomposition of the BBT posterior predictive to see what hypotheses caused the biggest differences

def xs_to_str(xs):
    out = ''
    for x in xs:
        out += str(x[0]) + ' '
    out = out.strip()
    return out

def str_to_xs(xs_str):
    xs = []
    xs_list = xs_str.strip().split()
    xs = [[x] for x in xs_list]
    return xs

def load_model(fn_out_model):
    print('Loading model:',fn_out_model,'on',DEVICE)
    checkpoint = torch.load(fn_out_model, map_location=DEVICE, weights_only=False)
    batch_size = checkpoint['batch_size']
    nets_state_dict = checkpoint['nets_state_dict']
    input_size = checkpoint['langs']['input'].n_symbols
    output_size = checkpoint['langs']['output'].n_symbols
    emb_size = checkpoint['emb_size']
    dropout_p = checkpoint['dropout']
    ff_mult = checkpoint['ff_mult']
    myact = checkpoint['activation']
    nlayers_encoder = checkpoint['nlayers_encoder']
    nlayers_decoder = checkpoint['nlayers_decoder']
    langs = checkpoint['langs']
    net = BBT(emb_size, input_size, output_size,
        langs['input'].PAD_idx, langs['output'].PAD_idx,
        nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
        dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)        
    net.load_state_dict(nets_state_dict)
    net = net.to(device=DEVICE)
    return net, langs

def set_fontsize(ax,fs=24):
    ax.tick_params(axis='both', labelsize=fs)

def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

def compare_w_sparse_decomp(max_h_compare=10):
    # Compare Bayesian model and BBT in terms of sparse hypothesis decomposition
    # 
    # Input
    # max_h_compare : maximum number of hypotheses to use for comparison
    for i in range(N): # for each episode
        S = list_S_unique_xs[i]        
        xs = unique_xs[i]
        R = responses_xs[i]
        x_eval = R.keys()
        x_eval = [int(x) for x in x_eval]
        x_eval = sorted(x_eval)
        assert(x_eval==xeval_batch)
        human_pred = np.array([R[str(k)] for k in x_eval]).flatten()

        # get the model outputs based on probability of "yes"
        batch = dat.make_bbt_batch([S], langs_ftune)
        batch = dat.set_batch_to_device(batch)
        E_ftune = meval.batch_acc(batch, net_ftune, langs_ftune, max_length_eval, eval_type='sample_prob_yes')
        batch = dat.make_bbt_batch([S], langs_base)
        batch = dat.set_batch_to_device(batch)
        E_base = meval.batch_acc(batch, net_base, langs_base, max_length_eval, eval_type='sample_prob_yes')
        
        # do the sparse decomposition for each of BBT's posteriors
        samples_pred_ftune = batch['list_samples']
        samples_pred_ftune[0]['p_yes'] = E_ftune['p_yes']        
        struct_compare = meval.make_sparse_approx(samples_pred_ftune)
        struct_compare = struct_compare[0] # since we only have one episode in batch
        if len(struct_compare)>max_h_compare: struct_compare=struct_compare[:max_h_compare]
        for rank,s in enumerate(struct_compare): # for each sparse hypothesis

            hl = s['h_nonzero'] # to highlight
            h_name = s['h_name'] # hypothesis name
            fn_out = fig_dir+'/model_compare_X='+str(xs)+'; rank='+str(rank)+'; '+h_name+'.png'
            mytitle = 'X='+str(xs)+'; '+h_name+'; sparse='+str(s['post_bbt_sparse'])+'vs. bayes='+str(s['post_bayes'])
        
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
            fig.set_figwidth(18)
            fig.set_figheight(20)
            
            fs = 18
            # fs = 14
            ## Human data
            oracle.subplot_predictions(ax1,x_eval,human_pred,highlight=hl)
            ax1.set_title(mytitle,fontsize=fs)
            ax1.set_ylabel('Human', fontsize=fs)
            set_fontsize(ax1)
            
            ## Tenenbaum model
            xs_int = [int(x) for x in xs.split()]
            mypred_bayes = oracle.bayesian_predictions(x_eval, xs_int, H_all, log_prior_all)
            mypred_bayes = mypred_bayes.flatten()
            oracle.subplot_predictions(ax2,x_eval,mypred_bayes,highlight=hl)
            ax2.set_ylabel('Bayes', fontsize=fs)
            set_fontsize(ax2)

            ## Base model
            mypred_base_model = E_base['p_yes'].flatten()
            oracle.subplot_predictions(ax3,x_eval,mypred_base_model,highlight=hl)        
            ax3.set_ylabel('Distilled only', fontsize=fs)
            # ax3.set_ylabel('prob. of membership', fontsize=fs)
            set_fontsize(ax3)

            ## Fine-tuned model
            mypred_ftune_model = E_ftune['p_yes'].flatten()
            oracle.subplot_predictions(ax4,x_eval,mypred_ftune_model,highlight=hl)        
            ax4.set_ylabel('BBT', fontsize=fs)
            # ax4.set_ylabel('prob. of membership', fontsize=fs)
            set_fontsize(ax4)

            if do_make_figures: plt.savefig(fn_out)

if __name__ == '__main__':

    mytype = 'test' # val or test
    fn_base = 'net_ngame_30M_base_paired.pt'
    fn_ftune = 'net_ngame_30M_bbt_top.pt'
    fn_scratch = 'net_ngame_30M_scratch_top.pt'
    dir_model = 'out_models'
    do_sparse_decomp = False
    do_make_figures = True
    max_length_eval = 1 # max length while generating, does not need to be changed

    ## Compare alt models
    do_compare_alt_models = False
    fn_alt_fit_prior = 'out_models_alternative/bng_tuned_prior_sigma0.pt'
    fn_alt_fit_full  = 'out_models_alternative/bng_tuned_full_sigma0.pt'

    assert(mytype in ['val','test'])
    fig_dir = 'figs_ngame_trials_'+mytype
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if mytype=='val':
        _, D_val = dat.get_dataset('number_game_human')
    else:
        _, D_val = dat.get_dataset('number_game_human_test')
    list_S = [S for S in D_val]
    unique_xs = sorted(list(set([xs_to_str(S['xs']) for S in list_S])))
    N = len(unique_xs)
    responses_xs = []

    # set up Bayesian model
    H_all,log_prior_all,H_names = oracle.generate_hypotheses(oracle.mylambda_default)
    
    # set up human data 
    # for each unique support set, get the probability of extending each number y
    for xs in unique_xs:
        S_sel = [S for S in list_S if xs_to_str(S['xs'])==xs]
        R = {} # reponse for each query
        for S in S_sel:
            xq = S['xq']
            assert(all([x==xq[0] for x in xq])) # check that all queries are the same
            xq = str(xq[0][0])
            yq = S['yq']
            p_yes = np.mean([y==['yes'] for y in yq])
            R[xq] = p_yes
        responses_xs.append(R)

    # set up models
    net_base, langs_base = load_model(dir_model+'/'+fn_base)
    net_ftune, langs_ftune = load_model(dir_model+'/'+fn_ftune)
    net_scratch, langs_scratch = load_model(dir_model+'/'+fn_scratch)
    list_S_unique_xs = []
    xeval_batch = list(range(1,101))
    for i in range(N):
        xs = str_to_xs(unique_xs[i])
        ys = [['yes'] for x in xs]
        xq = [[str(j)] for j in xeval_batch]
        yq = [['no'] for x in xq]
        S = dat.bundle_bbt_episode(xs,ys,xq,yq,myhash='')
        list_S_unique_xs.append(S)

    # set up alternative models
    if do_compare_alt_models:
        import fit_bayes_number_game as fit
        print('Loading',fn_alt_fit_prior)
        net_prior_tuned_best = fit.BayesNumGameOptPrior('bayes')
        fit.load_model(fn_alt_fit_prior,net_prior_tuned_best)
        fit.describe_model(net_prior_tuned_best)
        print('Loading',fn_alt_fit_full)
        net_full_tuned_best = fit.BayesNumGameOptFull('bayes')
        fit.load_model(fn_alt_fit_full,net_full_tuned_best)
        fit.describe_model(net_full_tuned_best)

    model_fit_r = {'bayes':[],'base':[],'ftune':[],'scratch':[], 'bayes_vs_base':[]}
    model_fit_rmse = {'bayes':[],'base':[],'ftune':[],'scratch':[], 'bayes_vs_base':[]}
    model_fit_raw = {'human':[],'bayes':[],'base':[],'ftune':[],'scratch':[],'alt_fit_prior':[],'alt_fit_full':[]}
    for i in range(N):
        S = list_S_unique_xs[i]        
        xs = unique_xs[i]
        R = responses_xs[i]
        x_eval = R.keys()
        x_eval = [int(x) for x in x_eval]
        x_eval = sorted(x_eval)
        assert(x_eval==xeval_batch)
        human_pred = np.array([R[str(k)] for k in x_eval]).flatten()
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig.set_figwidth(18)
        fig.set_figheight(20)
        
        fs = 14
        ## Human data
        oracle.subplot_predictions(ax1,x_eval,human_pred)
        model_fit_raw['human'].append(human_pred)
        ax1.set_ylabel('Human, X='+str(xs), fontsize=fs)
        # ax1.set_ylabel('prob. of membership', fontsize=fs)
        set_fontsize(ax1)
        
        ## Tenenbaum model
        xs_int = [int(x) for x in xs.split()]
        mypred_bayes = oracle.bayesian_predictions(x_eval, xs_int, H_all, log_prior_all)
        mypred_bayes = mypred_bayes.flatten()
        r_bayes = round(pearsonr(human_pred,mypred_bayes)[0],2)
        rmse_bayes = round(rmse(human_pred,mypred_bayes),3)
        model_fit_r['bayes'].append(r_bayes)
        model_fit_rmse['bayes'].append(rmse_bayes)
        model_fit_raw['bayes'].append(mypred_bayes)
        
        oracle.subplot_predictions(ax2,x_eval,mypred_bayes)
        ax2.set_ylabel('Bayes, r='+str(r_bayes)+', rmse='+str(rmse_bayes), fontsize=fs)
        # ax2.set_ylabel('prob. of membership', fontsize=fs)
        set_fontsize(ax2)

        ## Base model
        batch = dat.make_bbt_batch([S], langs_base)
        batch = dat.set_batch_to_device(batch)
        E = meval.batch_acc(batch, net_base, langs_base, max_length_eval, eval_type='sample_prob_yes')
        mypred_base_model = E['p_yes'].flatten()
        r_base = round(pearsonr(human_pred,mypred_base_model)[0],2)
        rmse_base = round(rmse(human_pred,mypred_base_model),3)
        r_mimic = round(pearsonr(mypred_bayes,mypred_base_model)[0],2)
        model_fit_r['base'].append(r_base)
        model_fit_raw['base'].append(mypred_base_model)
        model_fit_rmse['base'].append(rmse_base)
        model_fit_r['bayes_vs_base'].append(r_mimic)
        oracle.subplot_predictions(ax3,x_eval,mypred_base_model)        
        ax3.set_ylabel('Distilled only, r='+str(r_base)+', rmse='+str(rmse_base), fontsize=fs)
        # ax3.set_ylabel('prob. of membership', fontsize=fs)
        set_fontsize(ax3)

        ## Fine-tuned model
        batch = dat.make_bbt_batch([S], langs_ftune)
        batch = dat.set_batch_to_device(batch)
        E = meval.batch_acc(batch, net_ftune, langs_ftune, max_length_eval, eval_type='sample_prob_yes')
        mypred_ftune_model = E['p_yes'].flatten()
        r_ftune = round(pearsonr(human_pred,mypred_ftune_model)[0],2)
        rmse_ftune = round(rmse(human_pred,mypred_ftune_model),3)
        model_fit_r['ftune'].append(r_ftune)
        model_fit_rmse['ftune'].append(rmse_ftune)
        model_fit_raw['ftune'].append(mypred_ftune_model)
        oracle.subplot_predictions(ax4,x_eval,mypred_ftune_model)        
        ax4.set_ylabel('BBT, r='+str(r_ftune)+', rmse='+str(rmse_ftune), fontsize=fs)
        # ax4.set_ylabel('prob. of membership', fontsize=fs)
        set_fontsize(ax4)

        ## Scratch model
        batch = dat.make_bbt_batch([S], langs_scratch)
        batch = dat.set_batch_to_device(batch)
        E = meval.batch_acc(batch, net_scratch, langs_scratch, max_length_eval, eval_type='sample_prob_yes')
        mypred_scratch_model = E['p_yes'].flatten()
        r_scratch = round(pearsonr(human_pred,mypred_scratch_model)[0],2)
        rmse_scratch = round(rmse(human_pred,mypred_scratch_model),3)
        model_fit_r['scratch'].append(r_scratch)
        model_fit_rmse['scratch'].append(rmse_scratch)
        model_fit_raw['scratch'].append(mypred_scratch_model)

        # Alternative models
        if do_compare_alt_models:
            mypred_alt_fit_prior = net_prior_tuned_best.pp(fit.convert_h_list_to_tensor(xs_int).view(1,-1))
            mypred_alt_fit_prior = mypred_alt_fit_prior.detach().cpu().numpy().flatten()[1:]
            model_fit_raw['alt_fit_prior'].append(mypred_alt_fit_prior)

            mypred_alt_fit_full = net_full_tuned_best.pp(fit.convert_h_list_to_tensor(xs_int).view(1,-1))
            mypred_alt_fit_full = mypred_alt_fit_full.detach().cpu().numpy().flatten()[1:]
            model_fit_raw['alt_fit_full'].append(mypred_alt_fit_full)

        if do_make_figures: plt.savefig(fig_dir+'/model_compare_X='+str(xs)+'.png')

    print(mytype,'set')
    print('Summary of model fits (averaged over trials):')
    print('Correlation:')
    print('  mean r=',np.mean(model_fit_r['bayes']),'for Bayesian model')
    print('  mean r=',np.mean(model_fit_r['base']),'for base net')
    print('  mean r=',np.mean(model_fit_r['ftune']),'for bbt model')
    print('  mean r=',np.mean(model_fit_r['scratch']),'for scratch model')
    print('RMSE:')
    print('  mean rmse=',np.mean(model_fit_rmse['bayes']),'for Bayesian model')
    print('  mean rmse=',np.mean(model_fit_rmse['base']),'for base net')
    print('  mean rmse=',np.mean(model_fit_rmse['ftune']),'for bbt model')
    print('  mean rmse=',np.mean(model_fit_rmse['scratch']),'for scratch model')
    print("")
    print('  mean r=',np.mean(model_fit_r['bayes_vs_base']),'for Bayesian vs. base net (mimic test)')

    print("")
    print(mytype,'set')    
    model_fit_raw['human'] = np.hstack(model_fit_raw['human'])
    model_fit_raw['bayes'] = np.hstack(model_fit_raw['bayes'])
    model_fit_raw['base'] = np.hstack(model_fit_raw['base'])
    model_fit_raw['ftune'] = np.hstack(model_fit_raw['ftune'])
    model_fit_raw['scratch'] = np.hstack(model_fit_raw['scratch'])
    if do_compare_alt_models:
        model_fit_raw['alt_fit_prior'] = np.hstack(model_fit_raw['alt_fit_prior'])
        model_fit_raw['alt_fit_full'] = np.hstack(model_fit_raw['alt_fit_full'])
    print('Overall fits (aggregate across all prob. predictions; N=',model_fit_raw['human'].shape,'):')
    print('Correlation:')
    print('  r=',round(pearsonr(model_fit_raw['human'],model_fit_raw['bayes'])[0],2),'for Bayesian model')
    print('  r=',round(pearsonr(model_fit_raw['human'],model_fit_raw['base'])[0],2),'for base net')
    print('  r=',round(pearsonr(model_fit_raw['human'],model_fit_raw['ftune'])[0],2),'for BBT')
    print('  r=',round(pearsonr(model_fit_raw['human'],model_fit_raw['scratch'])[0],2),'for scratch')
    print('  r=',round(pearsonr(model_fit_raw['bayes'],model_fit_raw['base'])[0],4),'for Bayesian vs. base net (mimic test)')
    if do_compare_alt_models:
        print('  r=',round(pearsonr(model_fit_raw['human'],model_fit_raw['alt_fit_prior'])[0],2),'for Bayesian model (fit prior)')
        print('  r=',round(pearsonr(model_fit_raw['human'],model_fit_raw['alt_fit_full'])[0],2),'for Bayesian model (fit full)')
    print('RMSE:')
    print('  rmse=',round(rmse(model_fit_raw['human'],model_fit_raw['bayes']),3),'for Bayesian model')
    print('  rmse=',round(rmse(model_fit_raw['human'],model_fit_raw['base']),3),'for base net')
    print('  rmse=',round(rmse(model_fit_raw['human'],model_fit_raw['ftune']),3),'for BBT')
    print('  rmse=',round(rmse(model_fit_raw['human'],model_fit_raw['scratch']),3),'for scratch')
    print('  rmse=',round(rmse(model_fit_raw['bayes'],model_fit_raw['base']),3),'for Bayesian vs. base net (mimic test)')
    if do_compare_alt_models:
        print('  rmse=',round(rmse(model_fit_raw['human'],model_fit_raw['alt_fit_prior']),3),'for Bayesian model (fit prior)')
        print('  rmse=',round(rmse(model_fit_raw['human'],model_fit_raw['alt_fit_full']),3),'for Bayesian model (fit full)')


    if do_sparse_decomp: compare_w_sparse_decomp()

    # plt.figure()
    # plt.scatter(model_fit_raw['ftune'],model_fit_raw['human'],alpha=0.2)
    # plt.title('human vs. net (fine-tuned)')
    # plt.xlabel('model prob')
    # plt.ylabel('human prob')

    # plt.figure()
    # plt.scatter(model_fit_raw['bayes'],model_fit_raw['base'],alpha=0.2)
    # plt.title('bayes vs. mml (mimic test)')
    # plt.xlabel('model prob')
    # plt.ylabel('model prob')

    # plt.show()