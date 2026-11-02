import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import numpy as np
from bayes_fol_concepts import eval_hyp, split_set_by_obj_sep, x_to_fdata, fdata_to_x_long, fdata_to_x_short, get_format_x, bayesian_predictions, cache_LOT_hypotheses
from model import BBT
from copy import deepcopy
import datasets as dat
from sparse_pp_approx import flat, fit_sparse_pp_approx_set_concepts
from train_utils import seed_all, extract, assert_consist_langs, display_input_output, get_best_device
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, RegularPolygon
from matplotlib.offsetbox import OffsetImage, DrawingArea, AnnotationBbox
import eval as meval
from scipy.stats import pearsonr
import pickle
DEVICE = get_best_device()

# For each episode of human data, compare BBT with the alternative models.
#  if do_sparse_decomp, we do a decomposition of the BBT posterior predictive to see what hypotheses caused the biggest differences

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

def set_fontsize(ax,fs):
    ax.tick_params(axis='both', labelsize=fs)

def simpley(s):
        return s.replace('y2','y').replace('y3','y').replace('y4','y').replace('y5','y').replace('y6','y').replace('y7','y').replace('y8','y')

def abbv_hyp(s):
    s = s.replace('lambda (S)','L_S')
    s = s.replace('lambda (x)','L_x')
    s = s.replace('lambda S','L_S')
    s = s.replace('lambda x','L_x')
    return s

def num2str(x):
    x = str(x)
    if len(x)==1:
        return '00'+x
    if len(x)==2:
       return '0'+x
    return x

def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

def compare_w_sparse_decomp(max_h_compare=3):
    # Compare models in terms of sparse hypothesis decomposition
    # 
    # Input
    # max_h_compare : maximum number of hypotheses to use for comparison
    seed_all(0)
    with open('data/set_concepts_viz/hyp_total_dict.pkl', 'rb') as f:
        list_H = pickle.load(f)
        list_H_objects = list(list_H.values())

    print("Comparing models with sparse decomposition")
    for idx, hyp_gt in enumerate(episodes_by_hyp.keys()):
        if idx % 10==0: print('episode',idx,'of',len(episodes_by_hyp))             
        list_E = list(episodes_by_hyp[hyp_gt])
        list_ns = [len(remove_empty(E['xs'])) for E in list_E]
        list_E = [E for _, E in sorted(zip(list_ns, list_E), key=lambda t: t[0])]
        list_ns = sorted(list_ns)
        xq = [S['xq'][0] for S in list_E]
        xs = remove_empty(list_E[0]['xs'])
        assert(len(xs)==my_ns_range[0])
        objects_flat = flat([[s + '(set'+str(idx)+')' for s in split_set_by_obj_sep(myset)] for idx,myset in enumerate(xq)])
        yq_gold_flat = flat([S['yq_gold'] for S in list_E])
        
        human_pred = model_fit_raw['human'][idx]        
        mypred_base_model = model_fit_raw['base'][idx]
        mypred_ftune_model = model_fit_raw['ftune'][idx]

        xq = [S['xq'][0] for S in list_E]
        yq_gold = [S['yq_gold'] for S in list_E]
        
        def f_gold_format_long(query_set_as_object):
            # Given a query set (in object form), return a function that makes a gold label for an object in that set.
            # input
            #  query_set_as_object : list of lotlib objects, each an object int he set
            query_set_x = fdata_to_x_long(query_set_as_object) # convert each objet to list of symbols, with different objects separated as OBJ_SEP
            query_set_x_as_list = [item.strip().split(' ') for item in ' '.join(query_set_x).split('OBJ_SEP')] # set is now a list of lists, one for list of each object
            myidx = xq.index(query_set_x) # which episode has this query set? 
            query_set_labels = [mylab=='yes' for mylab in yq_gold[myidx]] # get the ground truth labels for this query set (as list of bool)
            return lambda myobj : query_set_labels[query_set_x_as_list.index(fdata_to_x_long([myobj]))] # return function that takes lotlib object and returns its label
        
        def f_gold_format_short(query_set_as_object):
            # Given a query set (in object form), return a function that makes a gold label for an object in that set.
            # input
            #  query_set_as_object : list of lotlib objects, each an object in the set
            query_set_x = fdata_to_x_short(query_set_as_object) # convert each object to a string. thus, query_set_x is a list of strings
            query_set_x_as_list = [[item] for item in query_set_x] # one for list of each object (just wraps a string in a one element list)
            myidx = xq.index(query_set_x) # which episode has this query set? 
            query_set_labels = [mylab=='yes' for mylab in yq_gold[myidx]] # get the ground truth labels for this query set (as list of bool)
            return lambda myobj : query_set_labels[query_set_x_as_list.index(fdata_to_x_short([myobj]))] # return function that takes lotlib object and returns its label
        
        x_format = get_format_x(xq[0])
        if x_format == 'long':
            f_gold = f_gold_format_long
        else:
            f_gold = f_gold_format_short
        
        objects_flat = flat([[s + '(set'+str(idx)+')' for s in split_set_by_obj_sep(myset)] for idx,myset in enumerate(xq)])
        yq_gold_flat = flat(yq_gold)
        
        tbase = torch.tensor(np.stack((mypred_base_model,1.-mypred_base_model),axis=-1), device=DEVICE)
        tftune = torch.tensor(np.stack((mypred_ftune_model,1.-mypred_ftune_model),axis=-1), device=DEVICE)
        struct_w_base = fit_sparse_pp_approx_set_concepts(tbase, xq, list_H_objects+[f_gold], verbose=False)
        struct_w_ftune = fit_sparse_pp_approx_set_concepts(tftune, xq, list_H_objects+[f_gold], verbose=False)

        sorted_w_ftune = struct_w_ftune['sorted_w'][:max_h_compare]
        sorted_names_ftune = struct_w_ftune['sorted_names'][:max_h_compare]
        sorted_w_base_matched = [struct_w_base['sorted_w'][struct_w_base['sorted_names'].index(myname)] for myname in sorted_names_ftune]

        for rank in range(max_h_compare):            
            w_ftune_rank = round(sorted_w_ftune[rank].item(),3)
            if w_ftune_rank < 0.01: continue # skip if this hypothesis has very low weight in fine-tuned model, since it won't be meaningful to compare with base model
            h_name_rank = simpley(sorted_names_ftune[rank])
            if 'f_gold' in h_name_rank: h_name_rank = 'GOLD'
            w_base_rank = round(sorted_w_base_matched[rank].item(),3)
            if h_name_rank == 'GOLD':
                h_pred = flat([eval_hyp(myset,f_gold) for myset in xq])                
            else:
                h_pred = flat([eval_hyp(myset,list_H[h_name_rank]) for myset in xq])
            assert all(y in ['yes', 'no'] for y in h_pred), "Each element must be 'yes' or 'no'"
            hl = [idx for idx,y in enumerate(h_pred) if y=='yes'] # convert to 1/0

            fn_out = fig_dir+'/'+num2str(idx)+'_'+abbv_hyp(hyp_gt)[:label_cut]+'; rank='+str(rank)+'; '+abbv_hyp(h_name_rank)[:label_cut]+'.png'
            mytitle = h_name_rank+'; ftune='+str(w_ftune_rank)+' vs. base='+str(w_base_rank)

            if inc_exact_bayes:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
                fig.set_figwidth(18)
                fig.set_figheight(26)
            else:
                fig, (ax1, ax3, ax4) = plt.subplots(3)
                fig.set_figwidth(18)
                fig.set_figheight(20)
            
            fs = 24
            ## Human data
            xrange = np.arange(len(objects_flat))
            ax1.bar(xrange, human_pred)
            ax1.bar(hl,human_pred[hl],color='orange')
            ax1.set_xticks(xrange)   
            set_shape_xticks(ax1, objects_flat, yq_gold_flat)     
            ax1.set_title(hyp_gt,fontsize=fs)
            ax1.set_ylabel('Human',fontsize=fs)       
            ax1.set_title(mytitle, fontsize=fs)
            ax1.set_ylim((0,1.2))
            ax1.set_yticks([0,0.5,1])            
            set_fontsize(ax1, fs)

            ## Bayesian ideal learner
            if inc_exact_bayes:
                mypred_bayes = model_fit_raw['bayes'][idx]
                ax2.bar(xrange, mypred_bayes)
                ax2.bar(hl, mypred_bayes[hl], color='orange')
                ax2.set_xticks(xrange)    
                ax2.set_ylabel('Bayesian', fontsize=fs)
                ax2.set_ylim((0,1.2))
                ax2.set_yticks([0,0.5,1])
                set_shape_xticks(ax2, objects_flat, yq_gold_flat)
                set_fontsize(ax2, fs)

            ## Base model
            ax3.bar(xrange, mypred_base_model)
            ax3.bar(hl,mypred_base_model[hl],color='orange')
            ax3.set_xticks(xrange)    
            set_shape_xticks(ax3, objects_flat, yq_gold_flat)
            ax3.set_ylabel('Distilled only', fontsize=fs)
            ax3.set_ylim((0,1.2))
            ax3 .set_yticks([0,0.5,1])
            set_fontsize(ax3, fs)

            ## Fine-tuned model
            ax4.bar(xrange, mypred_ftune_model)
            ax4.bar(hl,mypred_ftune_model[hl],color='orange')
            ax4.set_xticks(xrange)
            set_shape_xticks(ax4, objects_flat, yq_gold_flat)
            # ax4.set_xticklabels(objects_flat, rotation=75, fontsize=fs)
            ax4.set_ylabel('BBT', fontsize=fs)
            ax4.set_ylim((0,1.2))
            ax4.set_yticks([0,0.5,1])
            set_fontsize(ax4, fs)

            plt.savefig(fn_out)

def remove_empty(lst):
    return [x for x in lst if x != []]

def draw_shape_icon(ax, x_pos, y_pos, shape, color, size, max_pixel=20, highlight=False):
    #Draw a single shape icon at the given data coordinates.
    color_map = {
        'yellow': '#FFD700',
        'green':  '#2ECC71',
        'blue':   '#3498DB',
    }
    fc = color_map.get(color, color)

    # Scale factor based on size (1=small, 2=medium, 3=large)
    assert(size in [1, 2, 3])
    scale = [0.4, 0.8, 1.3][size-1]
    icon_size = max_pixel * scale

    da = DrawingArea(max_pixel * 1.5, max_pixel * 1.5, 0, 0)
    cx, cy = max_pixel * 0.75, max_pixel * 0.75  # center of drawing area

    if shape == 'circle':
        patch = mpatches.Circle((cx, cy), radius=icon_size / 2, fc=fc, ec='black', lw=1)
    elif shape == 'rectangle':
        half = icon_size / 2
        patch = mpatches.FancyBboxPatch((cx - half, cy - half), icon_size, icon_size,
                                         boxstyle="square,pad=0", fc=fc, ec='black', lw=1)
    elif shape == 'triangle':
        half = icon_size / 2
        verts = [
            (cx, cy + half),           # top
            (cx - half, cy - half),    # bottom-left
            (cx + half, cy - half),    # bottom-right
            (cx, cy + half),           # close
        ]
        patch = mpatches.Polygon(verts, closed=True, fc=fc, ec='black', lw=1)
    else:
        assert False, "Unknown shape: " + shape

    da.add_artist(patch)
    ab = AnnotationBbox(da, (x_pos, y_pos),
                        xycoords=('data', 'axes fraction'),
                        box_alignment=(0.5, 1.0),
                        frameon=False,
                        pad=0)
    ax.add_artist(ab)
    if highlight:
        ax.annotate('*', xy=(x_pos, y_pos), xycoords=('data', 'axes fraction'),
                    xytext=(0, -max_pixel * 2.5), textcoords='offset pixels',
                    ha='center', va='top', fontsize=26, color='black', fontweight='bold')


def draw_set_summary(objects_flat, yq_gold_flat):
    """Display each set's objects as a horizontal line of shape icons.

    One row per set; within each row the objects are evenly spaced left-to-right.
    Objects in the ground-truth concept are highlighted with *.

    Parameters
    ----------
    ax : matplotlib Axes  (should have equal aspect and no decorations)
    objects_flat : list of str  (same format as set_shape_xticks)
    yq_gold_flat : list of str, 'yes' or 'no'
    """
    import re
    from collections import defaultdict

    # Group by set index, preserving within-set order
    sets = defaultdict(list)   # set_idx -> [(label, in_concept), ...]
    for label, gold in zip(objects_flat, yq_gold_flat):
        m = re.search(r'\(set(\d+)\)', label)
        idx = int(m.group(1)) if m else 0
        sets[idx].append((label, gold == 'yes'))

    set_ids = sorted(sets.keys())
    n_sets = len(set_ids)
    max_n = max(len(v) for v in sets.values())

    _, (ax) = plt.subplots(1, figsize=(max(4, max_n * 0.6), max(1.5, n_sets * 1.1)))

    ax.set_xlim(-0.5, max_n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add set labels on the left as text annotations
    for row, set_idx in enumerate(set_ids):
        y_frac = 1.0 - (row + 0.5) / n_sets
        ax.text(-0.5, y_frac, f'set {set_idx}', ha='right', va='center',
                fontsize=9, transform=ax.get_yaxis_transform())

    for row, set_idx in enumerate(set_ids):
        # map row index to axes-fraction y (top row = near 1, bottom = near 0)
        y_frac = 1.0 - (row + 0.5) / n_sets
        items = sets[set_idx]
        for col, (label, highlight) in enumerate(items):
            clean = re.sub(r'\(set\d+\)', '', label).strip()
            parts = clean.split()
            shape = color = size = None
            for p in parts:
                if p in ('triangle', 'circle', 'rectangle'):
                    shape = p
                elif p in ('yellow', 'green', 'blue'):
                    color = p
                elif p.startswith('size_'):
                    size = int(p.split('_')[1])
            draw_shape_icon(ax, col, y_frac, shape, color, size,
                            highlight=highlight)


def set_shape_xticks(ax, objects_flat, yq_gold_flat, y_offset=-0.02):
    # """Replace text x-tick labels with visual shape icons.
    
    # Parameters
    # ----------
    # ax : matplotlib Axes
    # objects_flat : list of str
    #     Labels like 'triangle yellow size_1(set0)' or 'circle blue size_3(set2)'
    # yq_gold_flat : list of str, either 'yes' or 'no'
    # y_offset : float
    #     Vertical position in axes fraction (negative = below axis)
    # """
    import re
    ax.set_xticks(range(len(objects_flat)))
    ax.set_xticklabels(['' for _ in objects_flat])  # clear text labels

    # Parse set indices to find boundaries between sets
    set_indices = []
    for label in objects_flat:
        m = re.search(r'\(set(\d+)\)', label)
        set_indices.append(int(m.group(1)) if m else -1)

    for i, label in enumerate(objects_flat):
        # Strip the (setN) suffix
        clean = re.sub(r'\(set\d+\)', '', label).strip()
        parts = clean.split()
        for p in parts:
            if p in ('triangle', 'circle', 'rectangle'):
                shape = p
            elif p in ('yellow', 'green', 'blue'):
                color = p
            elif p.startswith('size_'):
                size = int(p.split('_')[1])
                assert(size in [1, 2, 3])
            else:
                assert False, "Unknown part in label: " + p
        in_gt_concept = yq_gold_flat[i] == 'yes'
        draw_shape_icon(ax, i, y_offset, shape, color, size, highlight=in_gt_concept)

    # Draw a short vertical tick on the x-axis at each set boundary
    for i in range(1, len(set_indices)):
        if set_indices[i] != set_indices[i - 1]:
            boundary = i - 0.5
            line = plt.Line2D([boundary, boundary], [-0.1, 0.0],
                              color='black', linewidth=1.5,
                              clip_on=False)    
            ax.add_artist(line)

    # Add some padding at the bottom so icons aren't clipped
    ax.tick_params(axis='x', which='both', length=0, pad=25)

if __name__ == '__main__':
    inc_exact_bayes = False
    batch_size = 1
    my_ns_range = [17,24] # use trials that have study sets with this many object-lists (inclusive). corresponds to trial 18-25
    mytype = 'human' # only include human data
    fn_base = 'net_set_30M_base_paired.pt'
    fn_ftune = 'net_set_30M_bbt_top.pt'
    fn_scratch = 'net_set_30M_scratch_top.pt'

    fig_dir = 'figs_set_trials_test'
    dir_model = 'out_models'
    do_sparse_decomp = False
    do_make_figures = True
    axis_labels = False
    label_cut = 80 # don't include more characters than this is a hypothsesis name
    max_length_eval = 6 # max length while generating, does not need to be changed

    # Bayesian ideal learner parameters
    bayes_model_type = 'uniform' #'temperature'
    ll_type = 'hard' # 'soft'

    assert(mytype in ['human'])
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    _, D_val = dat.get_dataset('set_concepts_human_test')
    list_S = [D_val[i] for i in range(len(D_val))]
    list_S = [E for E in list_S if my_ns_range[0] <= len(remove_empty(E['xs'])) <= my_ns_range[1]]
    # check that for each E['xq'] in list_S, which itself is a list, has identical elements
    for S in list_S:
        assert(all([x==S['xq'][0] for x in S['xq']])) # all queries are the same within episode
        S['yq_gold'] = S['aux']['h_str'].split('\n')[1].lstrip('gold_labels=[').rstrip(']').replace("'","").split(', ')
    episodes_by_hyp = {}
    list_hyp_gt = [E['aux']['h_str'].split('\n')[0].lstrip('h_str="').rstrip('"') for E in list_S]
    for hyp, ep in zip(list_hyp_gt, list_S):
        episodes_by_hyp.setdefault(hyp, []).append(ep)

    # set up models
    net_base, langs_base = load_model(dir_model+'/'+fn_base)
    net_ftune, langs_ftune = load_model(dir_model+'/'+fn_ftune)
    net_scratch, langs_scratch = load_model(dir_model+'/'+fn_scratch)
    # set up ideal learner
    if inc_exact_bayes:
        bayes_list_hypotheses, bayes_log_prior = cache_LOT_hypotheses(bayes_model_type)    

    model_fit_r = {'bayes':[],'bayes':[],'base':[],'ftune':[], 'scratch':[]}
    model_fit_rmse = {'bayes':[],'bayes':[],'base':[],'ftune':[], 'scratch':[]}
    model_fit_raw = {'human':[],'bayes':[],'base':[],'ftune':[], 'scratch':[]}
    seed_all(0)
    print('Comparing models on',len(episodes_by_hyp),'learning lists')
    for _idx, hyp_gt in enumerate(episodes_by_hyp.keys()):
        print('Episode',_idx,'of',len(episodes_by_hyp),':',hyp_gt)
        list_E = list(episodes_by_hyp[hyp_gt])
        list_ns = [len(remove_empty(E['xs'])) for E in list_E]
        list_E = [E for _, E in sorted(zip(list_ns, list_E), key=lambda t: t[0])]
        list_ns = sorted(list_ns)
        xq = [S['xq'][0] for S in list_E]
        xs = remove_empty(list_E[0]['xs'])
        ys = remove_empty(list_E[0]['ys'])
        assert(len(xs)==my_ns_range[0])
        yq_human = [np.array([[w == 'yes' for w in row] for row in S['yq']],dtype=float) for S in list_E]
        yq_human = [np.mean(myset, axis=0) for myset in yq_human]
        objects_flat = flat([[s + '(set'+str(idx)+')' for s in split_set_by_obj_sep(myset)] for idx,myset in enumerate(xq)])
        yq_gold_flat = flat([S['yq_gold'] for S in list_E])
        human_pred = flat(yq_human)
        human_pred = np.array(human_pred).flatten()

        if inc_exact_bayes:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
            fig.set_figwidth(18)
            fig.set_figheight(26)
        else:
            fig, (ax1, ax3, ax4) = plt.subplots(3)
            fig.set_figwidth(18)
            fig.set_figheight(20)
        
        fs = 24
        ## Human data
        xrange = np.arange(len(objects_flat))
        ax1.bar(xrange, human_pred)
        ax1.set_xticks(xrange)        
        if axis_labels:
            ax1.set_title(hyp_gt,fontsize=fs)       
            ax1.set_ylabel('prob. of membership', fontsize=fs)
        ax1.set_ylim((0,1.2))
        ax1.set_yticks([0,0.5,1])
        set_shape_xticks(ax1, objects_flat, yq_gold_flat)
        model_fit_raw['human'].append(human_pred)
        set_fontsize(ax1, fs)

        ## Bayesian ideal learner
        if inc_exact_bayes:
            yq_bayes, (best_hyp, best_log_post) = bayesian_predictions(xq, xs, ys, bayes_list_hypotheses, bayes_log_prior, ll_type=ll_type)
            mypred_bayes = np.array(flat(yq_bayes))
            r_bayes = pearsonr(human_pred,mypred_bayes)[0].item()
            rmse_bayes = rmse(human_pred,mypred_bayes)
            model_fit_r['bayes'].append(r_bayes)
            model_fit_rmse['bayes'].append(rmse_bayes)
            model_fit_raw['bayes'].append(mypred_bayes)
            ax2.bar(xrange, mypred_bayes)
            ax2.set_xticks(xrange)    
            if axis_labels:
                ax2.set_ylabel('Bayes('+ll_type+'), r='+str(round(r_bayes,2))+', rmse='+str(round(rmse_bayes,3)), fontsize=fs)
            ax2.set_ylim((0,1.2))
            ax2.set_yticks([0,0.5,1])
            if axis_labels:
                ax2.set_title(abbv_hyp(best_hyp)[:label_cut]+str(round(math.exp(best_log_post),2)), fontsize=fs)
            set_shape_xticks(ax2, objects_flat, yq_gold_flat)
            set_fontsize(ax2, fs)
            print('episode',_idx,': best Bayes hypothesis:')
            print('   ',best_hyp,'with posterior weight',round(math.exp(best_log_post),3))

        ## Base model
        list_E_prune = deepcopy(list_E)
        mask_is_object = np.zeros((len(list_E),max_length_eval), dtype=bool) # model outputs. which ones a real vs. padding
        for idx,E in enumerate(list_E_prune):
            E['xq'] = [E['xq'][0]] # because all of the queries are the same within an episode
            E['yq'] = [E['yq'][0]]
            E['xq_context']= [E['xq_context'][0]]
            mask_is_object[idx,:len(E['yq'][0])] = True
        batch = dat.make_bbt_batch(list_E_prune, langs_base)
        batch = dat.set_batch_to_device(batch)
        E = meval.batch_acc(batch, net_base, langs_base, max_length_eval, eval_type='sample_prob_yes')
        mypred_base_model = E['p_yes'][mask_is_object] # flattened list of model predictions for the objects, in same ordering as human_pred
        r_base = pearsonr(human_pred,mypred_base_model)[0].item()
        rmse_base = rmse(human_pred,mypred_base_model)
        model_fit_r['base'].append(r_base)
        model_fit_rmse['base'].append(rmse_base)
        model_fit_raw['base'].append(mypred_base_model)
        ax3.bar(xrange, mypred_base_model)
        ax3.set_xticks(xrange)
        if axis_labels:    
            ax3.set_ylabel('Distilled only, r='+str(round(r_base,2))+', rmse='+str(round(rmse_base,3)), fontsize=fs)
        ax3.set_ylim((0,1.2))
        ax3.set_yticks([0,0.5,1])
        set_shape_xticks(ax3, objects_flat, yq_gold_flat)
        # ax3.set_ylabel('prob. of membership', fontsize=fs)
        set_fontsize(ax3, fs)

        ## Fine-tuned model
        batch = dat.make_bbt_batch(list_E_prune, langs_ftune)
        batch = dat.set_batch_to_device(batch)
        E = meval.batch_acc(batch, net_ftune, langs_ftune, max_length_eval, eval_type='sample_prob_yes')
        mypred_ftune_model = E['p_yes'][mask_is_object]
        r_ftune = pearsonr(human_pred,mypred_ftune_model)[0].item()
        rmse_ftune = rmse(human_pred,mypred_ftune_model)
        model_fit_r['ftune'].append(r_ftune)
        model_fit_rmse['ftune'].append(rmse_ftune)
        model_fit_raw['ftune'].append(mypred_ftune_model)
        ax4.bar(xrange, mypred_ftune_model)
        ax4.set_xticks(xrange)
        ax4.set_ylim((0,1.2))
        ax4.set_yticks([0,0.5,1])
        set_shape_xticks(ax4, objects_flat, yq_gold_flat)
        if axis_labels:
            ax4.set_ylabel('BBT, r='+str(round(r_ftune,2))+', rmse='+str(round(rmse_ftune,3)), fontsize=fs)

        # Scratch-model (no plotting)
        batch = dat.make_bbt_batch(list_E_prune, langs_scratch)
        batch = dat.set_batch_to_device(batch)
        E = meval.batch_acc(batch, net_scratch, langs_scratch, max_length_eval, eval_type='sample_prob_yes')
        mypred_scratch_model = E['p_yes'][mask_is_object]
        r_scratch = pearsonr(human_pred,mypred_scratch_model)[0].item()
        rmse_scratch = rmse(human_pred,mypred_scratch_model)
        model_fit_r['scratch'].append(r_scratch)
        model_fit_rmse['scratch'].append(rmse_scratch)
        model_fit_raw['scratch'].append(mypred_scratch_model)

        # ax4.set_ylabel('prob. of membership', fontsize=fs)
        set_fontsize(ax4, fs)
        if do_make_figures:             
            plt.savefig(fig_dir+'/'+num2str(_idx)+'_'+abbv_hyp(hyp_gt)[:label_cut]+'.png')
            draw_set_summary(objects_flat, yq_gold_flat)
            plt.savefig(fig_dir+'/'+num2str(_idx)+'_data.pdf')

    print(mytype,'set')
    print('Summary of model fits (averaged over trials, note that NaN values are ignored):')    
    print(' Correlation:')
    if inc_exact_bayes: print('  mean r=',np.nanmean(model_fit_r['bayes']),'for Bayesian ideal learner, (n=',np.sum(~np.isnan(model_fit_r['bayes'])),')')
    print('  mean r=',np.nanmean(model_fit_r['base']),'for distillation only, (n=',np.sum(~np.isnan(model_fit_r['base'])),')')
    print('  mean r=',np.nanmean(model_fit_r['ftune']),'for BBT (n=',np.sum(~np.isnan(model_fit_r['ftune'])),')')
    print('  mean r=',np.nanmean(model_fit_r['scratch']),'for scratch model (n=',np.sum(~np.isnan(model_fit_r['scratch'])),')')
    print(' RMSE:')
    if inc_exact_bayes: print('  mean rmse=',np.nanmean(model_fit_rmse['bayes']),'for Bayesian ideal learner, (n=',np.sum(~np.isnan(model_fit_rmse['bayes'])),')')
    print('  mean rmse=',np.nanmean(model_fit_rmse['base']),'for distillation only, (n=',np.sum(~np.isnan(model_fit_rmse['base'])),')')
    print('  mean rmse=',np.nanmean(model_fit_rmse['ftune']),'for BBT (n=',np.sum(~np.isnan(model_fit_rmse['ftune'])),')')
    print('  mean rmse=',np.nanmean(model_fit_rmse['scratch']),'for scratch model (n=',np.sum(~np.isnan(model_fit_rmse['scratch'])),')')

    print("")
    print(mytype,'set')    
    v_prob_human = np.concatenate(model_fit_raw['human'])
    if inc_exact_bayes: v_prob_bayes = np.concatenate(model_fit_raw['bayes'])
    v_prob_base = np.concatenate(model_fit_raw['base'])
    v_prob_ftune = np.concatenate(model_fit_raw['ftune'])
    v_prob_scratch = np.concatenate(model_fit_raw['scratch'])
    print('Overall fits (aggregate across all prob. predictions; N=',v_prob_human.shape[0],'):')
    print( ' there is some randomness...')
    print('Correlation:')
    if inc_exact_bayes: print('  r=',round(pearsonr(v_prob_human,v_prob_bayes)[0],3),'for Bayesian ideal learner')
    print('  r=',round(pearsonr(v_prob_human,v_prob_base)[0],3),'for distillation only')
    print('  r=',round(pearsonr(v_prob_human,v_prob_ftune)[0],3),'for BBT')
    print('  r=',round(pearsonr(v_prob_human,v_prob_scratch)[0],3),'for scratch model')
    print('RMSE')
    if inc_exact_bayes: print('  rmse=',round(rmse(v_prob_human,v_prob_bayes),3),'for Bayesian ideal learner')
    print('  rmse=',round(rmse(v_prob_human,v_prob_base),3),'for distillation only')
    print('  rmse=',round(rmse(v_prob_human,v_prob_ftune),3),'for BBT')
    print('  rmse=',round(rmse(v_prob_human,v_prob_scratch),3),'for scratch model')

    if do_sparse_decomp: compare_w_sparse_decomp()