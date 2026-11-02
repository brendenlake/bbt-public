import os
import shutil
import glob
import random
import pickle
import math

from scipy.special import logsumexp
from train_utils import seed_all
import bayes_num_game
import torch
import numpy as np

def generate_num_game_dataset(nsamp_train,nsamp_val,nsamp_support=10,nsamp_query=20,mydir='data/num_game'):
    # Generate and write dataset to file
    # 
    # Input
    #  nsamp_train : number of training episodes
    #  nsamp_val : number of validation episodes
    #  nsamp_support: number of data points to sample as support
    #  mydir : output folder
    #
    # Output
    #   writes text files for each of the episodes
    #  
    if os.path.exists(mydir):
        raise ValueError('Attempting to create dataset in existing directory: ' + mydir)
    os.mkdir(mydir)
    os.mkdir(mydir+'/train')
    os.mkdir(mydir+'/val')
    print( "Generating validation episodes")
    samples = bayes_num_game.forward_predictive_model(nsamp_val, nsamp_support, nsamp_query)
    for s,sample in enumerate(samples): 
        sample = convert_nums_to_symbols(sample)
        episode_to_file(os.path.join(mydir,'val',num2str(s)+'.txt'),sample)
    print( "Generating training episodes")
    samples = bayes_num_game.forward_predictive_model(nsamp_train, nsamp_support, nsamp_query)
    for s,sample in enumerate(samples):
        sample = convert_nums_to_symbols(sample)
        episode_to_file(os.path.join(mydir,'train',num2str(s)+'.txt'),sample)

def generate_num_game_viz(nsamp_per_hyp, nsamp_per_interval=1,mydir='data/num_game_viz',max_ns=5,min_size_interval=3):
    # Generate episodes for a particular set of hypotheses for visualizing the embedding space
    #
    # Input
    #  nsamp_per_hyp : number of episodes to generate for each mathematical hypothesis
    #  nsamp_per_interval : number of episodes to sample from each chosen interval hypothesis
    #  mydir : output folder
    #  max_ns : maximum number of support points
    #  min_size_interval : minimum size of interval hypothesis to visualize
    if os.path.exists(mydir):
        raise ValueError('Attempting to create dataset in existing directory: ' + mydir)
    os.mkdir(mydir)
    H_numpy,_,H_names = bayes_num_game.generate_hypotheses(bayes_num_game.mylambda_default)
    size_each_h = np.sum(np.stack(H_numpy,axis=0),axis=1)

    # set up Bayesian model with fast inference
    from fit_bayes_number_game import BayesNumGameNet, convert_x_list_to_tensor
    BP_fast = BayesNumGameNet()
    def fast_posterior(h_target, x_as_list_of_ints):
        assert(all([isinstance(x,int) for x in x_as_list_of_ints]))
        data_t = convert_x_list_to_tensor(x_as_list_of_ints).view(1,-1)
        log_post = BP_fast.log_posterior(data_t)
        post = torch.exp(log_post).detach()
        return post

    hyp_even_odds = ['evens','odds']
    hyp_powers_of_n = [h for h in H_names if 'powers_of' in h]
    hyp_multiples_of_n = [h for h in H_names if 'mults_of' in h]
    hyp_ending_in_n = [h for h in H_names if ('ending_in' in h) and (h != 'ending_in_0')]
    hyp_math_highlight = hyp_even_odds + hyp_powers_of_n + hyp_multiples_of_n + hyp_ending_in_n
    nmath = len(hyp_math_highlight)
    hyp_int_highlight = [h for j,h in enumerate(H_names) if (',...,' in h) and (size_each_h[j] >= min_size_interval)]
    hyp_int_highlight = random.sample(hyp_int_highlight, nmath*nsamp_per_hyp)

    for myhyp in hyp_math_highlight:
        print( "\nGenerating episodes for hypothesis: " + myhyp)
        for s in range(nsamp_per_hyp):
            print(str(s), end=',')
            while True:
                n_support = random.randint(2,max_ns)
                sample = bayes_num_game.forward_model_special(n_support, myhyp)
                post = fast_posterior(myhyp,sample['xs'])
                max_hyp_index = post.argmax().item()
                max_hyp_name = H_names[max_hyp_index]
                if max_hyp_name == myhyp: # check if the most likely hypothesis is the target one
                    break
            sample = convert_nums_to_symbols(sample)
            episode_to_file(os.path.join(mydir,'hyp_'+myhyp+'_'+num2str(s)+'.txt'),sample)

    for j,myhyp in enumerate(hyp_int_highlight):
        print( "Generating episodes for interval",myhyp,'(',j,'of',len(hyp_int_highlight),')')
        for s in range(nsamp_per_interval):            
            while True:
                n_support = random.randint(2,max_ns)
                sample = bayes_num_game.forward_model_special(n_support, myhyp)
                post = fast_posterior(myhyp,sample['xs'])
                max_hyp_index = post.argmax().item()
                max_hyp_name = H_names[max_hyp_index]
                if ',...,' in max_hyp_name: # check if the most likely hypothesis is an interval one
                    break
            sample = convert_nums_to_symbols(sample)
            episode_to_file(os.path.join(mydir,'interval_'+myhyp+'_'+num2str(s)+'.txt'),sample)

# def generate_fol_concepts_dataset(nsamp_train,nsamp_val,nsamp_support=25,nsamp_query=10,mydir='data/set_concepts_hardll'):
#     # Input
#     #  nsamp_train : number of training episodes
#     #  nsamp_val : number of validation episodes
#     #  nsamp_support: number of data points to sample as support
#     #  mydir : output folder
#     assert os.path.exists(mydir)
#     os.mkdir(mydir+'/train')
#     os.mkdir(mydir+'/val')
#     print( "Generating validation episodes")
#     samples = bayes_fol_concepts.forward_predictive_model(nsamp_val, nsamp_support, nsamp_query)
#     for s,sample in enumerate(samples):
#         episode_to_file(os.path.join(mydir,'val',num2str(s)+'.txt'),sample)
#     print( "Generating training episodes")
#     samples = bayes_fol_concepts.forward_predictive_model(nsamp_train, nsamp_support, nsamp_query)
#     for s,sample in enumerate(samples):
#         episode_to_file(os.path.join(mydir,'train',num2str(s)+'.txt'),sample)

def generate_fol_concepts_uniform_dataset(nsamp_train,nsamp_val,nsamp_support=25,nsamp_query=10,mydir='data/set_concepts_uniform_hardll'):
    # This is verion of the FOL grammar with a flat prior
    # Input
    #  nsamp_train : number of training episodes
    #  nsamp_val : number of validation episodes
    #  nsamp_support: number of data points to sample as support
    #  mydir : output folder
    import bayes_fol_concepts as bfc_uniform
    bfc_uniform.grammar = bfc_uniform.set_grammar_weights_to_uniform(bfc_uniform.grammar)
    assert os.path.exists(mydir)
    os.mkdir(mydir+'/train')
    os.mkdir(mydir+'/val')
    print( "Generating validation episodes")
    samples = bfc_uniform.forward_predictive_model(nsamp_val, nsamp_support, nsamp_query)
    for s,sample in enumerate(samples):
        episode_to_file(os.path.join(mydir,'val',num2str(s)+'.txt'),sample)
    print( "Generating training episodes")
    samples = bfc_uniform.forward_predictive_model(nsamp_train, nsamp_support, nsamp_query)
    for s,sample in enumerate(samples):
        episode_to_file(os.path.join(mydir,'train',num2str(s)+'.txt'),sample)

# def generate_fol_concepts_temperature_dataset(nsamp_train,nsamp_val,nsamp_support=25,nsamp_query=10,nsamp_background=1000000,mydir='data/set_concepts_temperature'):
#     # This is verion of the FOL grammar with an approximation of the original prior
#     # Input
#     #  nsamp_train : number of training episodes
#     #  nsamp_val : number of validation episodes
#     #  nsamp_support: number of data points to sample as support
#     #  mydir : output folder
#     import bayes_fol_concepts as bfc
#     if os.path.exists(mydir):
#             raise ValueError('Attempting to create dataset in existing directory: ' + mydir)
#     os.mkdir(mydir)
#     os.mkdir(mydir+'/train')
#     os.mkdir(mydir+'/val')

#     grammar_fit = bfc.grammar
#     grammar_uniform = bfc.set_grammar_weights_to_uniform(grammar_fit) # proposal distribution
#     list_hyp = []
#     print('Sampling background hypotheses from uniform grammar')
#     for j in range(nsamp_background):
#         if j % 10000 == 0: print('  sampling',j,'of',nsamp_background)
#         myhyp = grammar_uniform.generate()
#         list_hyp.append(myhyp)
#     list_hyp_unique = list(set(list_hyp))
#     scores_hyp_unique = np.array([grammar_fit.log_probability(h) for h in list_hyp_unique])
#     scores_hyp_unique = scores_hyp_unique / bfc.temperature
#     log_weights = scores_hyp_unique - logsumexp(scores_hyp_unique)
#     weights = np.exp(log_weights) 
#     nh = len(weights)
#     idx_choices_train = np.random.choice(np.arange(nh), size=nsamp_train, replace=True, p=weights)
#     idx_choices_val = np.random.choice(np.arange(nh), size=nsamp_val, replace=True, p=weights)
    
#     print( "Generating validation episodes")
#     for s_idx in range(nsamp_val):
#         h = list_hyp_unique[idx_choices_val[s_idx]]
#         h_str = 'lambda S: lambda x: ' + str(h)
#         sample = bfc.forward_model_special_soft(nsamp_support, nsamp_query, eval(h_str))
#         sample['aux']={'h_str':h_str}
#         episode_to_file(os.path.join(mydir,'val',num2str(s_idx)+'.txt'),sample)
#     print( "Generating training episodes")
#     for s_idx in range(nsamp_train):
#         h = list_hyp_unique[idx_choices_train[s_idx]]
#         h_str = 'lambda S: lambda x: ' + str(h)
#         sample = bfc.forward_model_special_soft(nsamp_support, nsamp_query, eval(h_str))
#         sample['aux']={'h_str':h_str}
#         episode_to_file(os.path.join(mydir,'train',num2str(s_idx)+'.txt'),sample)

def h_is_more_complex(h, list_x_pats, list_y_pats):
    import bayes_fol_concepts as bfc_uniform
    # h : hypothesis
    # list_x_pats : list of sets to compute boolean outputs for
    # list_y_pats : list of all boolean patterns from other hypotheses
    h_str = str(h)
    quantifier_count = sum([h_str.count(q) for q in ['exists_(', 'forall_(', 'exists_exactly_one_(', 'exists_one_or_fewer_(']])
    has_connective = (' and_(' in h_str) or (' or_(' in h_str) or (' implies_(' in h_str) or (' iff_(' in h_str)
    is_complex = (quantifier_count >= 2) or (quantifier_count==1 and has_connective)
    my_bool_pats = [bfc_uniform.eval_hyp(myset,h) for myset in list_x_pats]
    if all([b=='yes' for b in [item for sublist in my_bool_pats for item in sublist]]): return False # reject trivial hypotheses that are always true
    if all([b=='no'  for b in [item for sublist in my_bool_pats for item in sublist]]): return False # reject trivial hypotheses that are always false
    is_unique = my_bool_pats not in list_y_pats
    return is_complex and is_unique

def generate_fol_concepts_viz(nsamp_per_hyp, mydir='data/set_concepts_viz', nsamp_support=24, nsamp_query=20, num_sets_yes=2, inc_complex=False):
    # Generate episodes for a particular set of hypotheses for visualizing the embedding space.
    # Then, generate an alternative set of hypotheses that are unique and more complex, for comparison.
    #
    # Input
    #  nsamp_per_hyp : number of episodes to generate for each hypothesis
    #  mydir : output folderç
    #  nsamp_support : number of support samples to draw
    #  nsamp_query : number of query samples to draw
    #  num_sets_yes_support : number of support/query sets that should have at least this many 'yes' in their output
    # inc_complex : if True, generate a second set of alternative hypotheses that are more complex than the original set
    #
    seed_all(0)
    import bayes_fol_concepts as bfc_uniform
    from itertools import combinations
    # bfc_uniform.grammar = bfc_uniform.grammar_add_exists_exactly_one(bfc_uniform.grammar)
    bfc_uniform.grammar = bfc_uniform.set_grammar_weights_to_uniform(bfc_uniform.grammar)
    if os.path.exists(mydir):
        raise ValueError('Attempting to create dataset in existing directory: ' + mydir)
    os.mkdir(mydir)
    os.mkdir(mydir+'/special')
    
    prefix = 'lambda S: lambda x: '
    hyp_colors = ['yellow_(x)', 'blue_(x)', 'green_(x)']
    hyp_sizes = ['size1_(x)', 'size2_(x)', 'size3_(x)']
    hyp_shapes = ['triangle_(x)', 'rectangle_(x)', 'circle_(x)']
    hyp_features = hyp_colors + hyp_sizes + hyp_shapes
    hyp_features_y2 = [f.replace('_(x)', '_(y2)') for f in hyp_features]

    def simpley(s):
        return s.replace('y2','y').replace('y3','y').replace('y4','y').replace('y5','y').replace('y6','y').replace('y7','y').replace('y8','y')
    def blankargs(s):
        s = simpley(s)
        for fx in hyp_features:
            fy = fx.replace('_(x)', '_(y)')
            s = s.replace(fx, 'f_(x)')
            s = s.replace(fy, 'f_(y)')
        return s
    
    hyp_bool = ['True','False']
    
    hyp_and  = ['and_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_sizes]
    hyp_and += ['and_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_shapes]
    hyp_and += ['and_(' + h + ', ' + h2 + ')' for h in hyp_sizes for h2 in hyp_shapes]
    hyp_or  = ['or_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_sizes]
    hyp_or += ['or_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_shapes]
    hyp_or += ['or_(' + h + ', ' + h2 + ')' for h in hyp_sizes for h2 in hyp_shapes]
    hyp_or += ['or_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_colors if (h != h2 and h < h2)]
    hyp_or += ['or_(' + h + ', ' + h2 + ')' for h in hyp_sizes for h2 in hyp_sizes if (h != h2 and h < h2)]
    hyp_or += ['or_(' + h + ', ' + h2 + ')' for h in hyp_shapes for h2 in hyp_shapes if (h != h2 and h < h2)] 
    hyp_implies = ['implies_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_sizes]
    hyp_implies += ['implies_(' + h + ', ' + h2 + ')' for h in hyp_sizes for h2 in hyp_colors]
    hyp_implies += ['implies_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_shapes]
    hyp_implies += ['implies_(' + h + ', ' + h2 + ')' for h in hyp_shapes for h2 in hyp_colors]
    hyp_implies += ['implies_(' + h + ', ' + h2 + ')' for h in hyp_sizes for h2 in hyp_shapes]
    hyp_implies += ['implies_(' + h + ', ' + h2 + ')' for h in hyp_shapes for h2 in hyp_sizes]
    hyp_iff = []
    hyp_iff = ['iff_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_sizes]
    hyp_iff += ['iff_(' + h + ', ' + h2 + ')' for h in hyp_colors for h2 in hyp_shapes]
    hyp_iff += ['iff_(' + h + ', ' + h2 + ')' for h in hyp_sizes for h2 in hyp_shapes]
    
    # EXISTS
    hyp_exists = ['exists_(lambda y2: '+f+'(x, y2), set_diff_custom_(S, x))' for f in ['size_geq_']]
    hyp_exists += ['exists_(lambda y2: '+f+'(y2, x), set_diff_custom_(S, x))' for f in ['size_geq_']]
    hyp_exists += ['exists_(lambda y2: '+f+'(x, y2), set_diff_custom_(S, x))' for f in ['shape_eq_','color_eq_','size_eq_']]
    # hyp_exists += ['exists_(lambda y2: '+f+', set_diff_custom_(S, x))' for f in hyp_features_y2]

    # EXISTS 1
    hyp_exists1 = [h_str.replace('exists_', 'exists_one_or_fewer_') for h_str in hyp_exists]

    # hyp_exists1 += [h_str.replace('exists_', 'exists_exactly_one_') for h_str in hyp_exists]
    # hyp_exists1 += ['exists_one_or_fewer_(lambda y2: '+f+', set_diff_custom_(S, x))' for f in hyp_features]
    # hyp_exists1 += ['exists_exactly_one_(lambda y2: '+f+', set_diff_custom_(S, x))' for f in hyp_features]

    # FOR ALL
    hyp_forall  = ['forall_(lambda y2: '+f+'(x, y2), S)' for f in ['size_geq_']]
    hyp_forall += ['forall_(lambda y2: '+f+'(y2, x), S)' for f in ['size_geq_']]
    hyp_forall += ['forall_(lambda y2: '+f+'(x, y2), S)' for f in ['shape_eq_','color_eq_','size_eq_']]
    # hyp_forall += ['forall_(lambda y2: '+f+', set_diff_custom_(S, x))' for f in hyp_features_y2] 

    all_hyp_types = [hyp_features, hyp_and, hyp_or, hyp_implies, hyp_iff, hyp_exists, hyp_forall, hyp_exists1, hyp_bool]
    hyp_total = []
    hyp_lens = []
    for my_hyp_types in all_hyp_types:
        hyp_total += my_hyp_types
        hyp_lens += len(my_hyp_types)*[float(len(my_hyp_types))] # mark each with the freq in each hypothesis type, for calibration later
    hyp_total = [simpley(h) for h in hyp_total]
    hyp_total = [prefix + h for h in hyp_total]
    hyp_total_dict = {h:None for h in hyp_total}
    
    mult = float(nsamp_per_hyp)*float(len(hyp_features)) # downsample relative to number of hypotheses for individual features
    hyp_freq_dict = {h:math.ceil(round(mult/hyp_lens[idx])) for idx,h in enumerate(hyp_total)}

    
    # Fill target hypotheses. Loop until every item in hyp_total_dict is filled (i.e., not None)
    max_attempts = 100000
    attempts = 0
    while any(v is None for v in hyp_total_dict.values()) and attempts < max_attempts:
        h = bfc_uniform.generate_hypothesis()
        sh = simpley(str(h))
        if sh in hyp_total_dict and hyp_total_dict[sh] is None:
            hyp_total_dict[sh] = h
        attempts += 1
        if attempts % 10000 == 0:
            remaining = sum(v is None for v in hyp_total_dict.values())
            print(f"Attempts: {attempts}, hypotheses left: {remaining}")
    if attempts >= max_attempts:
        missing = [h for h, v in hyp_total_dict.items() if v is None]
        print("Could not generate the following hypotheses:")
        for h in missing:
            print(h)
        assert False, "Maximum attempts reached, some hypotheses may not be generated."    

    # Generate boolean patterns for each hypothesis
    x_pats = [bfc_uniform.sample_set_objects(nobj_min=2) for _ in range(400)]
    y_pats = [[bfc_uniform.eval_hyp(myset,h) for myset in x_pats] for h in hyp_total_dict.values()]
    # Find and print pairs of y_pats elements that are identical, and the hypotheses they belong to
    y_pats_list = list(y_pats)
    hyp_list = list(hyp_total_dict.keys())
    for (i, j) in combinations(range(len(y_pats_list)), 2):
        if y_pats_list[i] == y_pats_list[j]:
            print(f"Identical y_pats for hypotheses:\n  {hyp_list[i]}\n  {hyp_list[j]}\n")

    # Generate episodes for each hypothesis
    for h_str, h in hyp_total_dict.items():
        my_freq = max(hyp_freq_dict[h_str],1)
        for j in range(my_freq):
            has_yes_in_ys = False
            has_yes_in_yq = False
            count = 0
            while not (has_yes_in_ys and has_yes_in_yq):
                sample = bfc_uniform.forward_model_special(nsamp_support, nsamp_query, h, nobj_min=2)
                has_yes_in_ys = sum(['yes' in y for y in sample['ys']]) >= num_sets_yes
                has_yes_in_yq = sum(['yes' in y for y in sample['yq']]) >= num_sets_yes
                if h_str.strip()=='lambda S: lambda x: False':
                    has_yes_in_ys = True
                    has_yes_in_yq = True
                    break # allow trivial False hypothesis to have no yes's
                count += 1
                if count > 1000:
                    print(f"Warning: Could not find a valid sample for hypothesis {h_str} after 1000 attempts.")
                    break
            if (has_yes_in_ys and has_yes_in_yq):
                episode_to_file(os.path.join(mydir, 'special', 'hyp_' + h_str.replace(' ','') + '_' + num2str(j) + '.txt'), sample)  

    # Generate episodes for each alternative hypothesis
    alt_hyp_sh = set()
    alt_count = 0
    hyp_alt_dict = {}
    blank_h_selected = set([blankargs(sh) for sh in hyp_total_dict.keys()])
    if inc_complex:
        hyp_iff_dict = {h: v for h, v in hyp_total_dict.items() if h in [prefix + simpley(f) for f in hyp_iff]}
        print('Generating ',len(hyp_iff_dict),'complex hypotheses')
        for yoked_h_str, _ in hyp_iff_dict.items():
            my_freq = max(hyp_freq_dict[yoked_h_str],1)
            h = yoked_h_str
            sh = simpley(str(yoked_h_str))
            blank_h_str = blankargs(sh)
            while (blank_h_str in blank_h_selected) or (blank_h_str in alt_hyp_sh) or (not h_is_more_complex(h,x_pats,y_pats)):
                h = bfc_uniform.generate_hypothesis()
                h_str = str(h)
                sh = simpley(str(h))
                blank_h_str = blankargs(sh)
            alt_hyp_sh.add(blank_h_str)
            hyp_alt_dict[sh] = h
            for j in range(my_freq):
                has_yes_in_ys = False
                has_yes_in_yq = False
                count = 0
                while not (has_yes_in_ys and has_yes_in_yq):
                    sample = bfc_uniform.forward_model_special(nsamp_support, nsamp_query, h, nobj_min=2)
                    has_yes_in_ys = sum(['yes' in y for y in sample['ys']]) >= num_sets_yes
                    has_yes_in_yq = sum(['yes' in y for y in sample['yq']]) >= num_sets_yes
                    count += 1
                    if count > 1000:
                        print(f"Warning: Could not find a valid sample for hypothesis {h_str} after 1000 attempts.")
                        break
                if (has_yes_in_ys and has_yes_in_yq):
                    episode_to_file(os.path.join(mydir, 'special', 'hyp_complex_' + str(alt_count) + '_' + num2str(j) + '.txt'), sample)
            alt_count += 1

    print('SELECTED HYPOTHESES')
    sorted_blank_h_selected = sorted(blank_h_selected, key=len)
    for elem in sorted_blank_h_selected:
        print(elem)
    print("")
    print('ALT COMPLEX HYPOTHESES')
    sorted_alt_hyp_sh = sorted(alt_hyp_sh, key=len)
    for elem in sorted_alt_hyp_sh:
        print(elem)
    
    with open(os.path.join(mydir, 'hyp_total_dict.pkl'), 'wb') as f:
        pickle.dump(hyp_total_dict | hyp_alt_dict, f)

def generate_num_game_special(mydir='data/num_game'):
    # Targeted trials from Tenenbaum article
    os.mkdir(mydir+'/special')
    xs = [16]
    S = {'xs':[[str(x)] for x in xs], 'ys':[['yes']]*len(xs),
     'xq':[[str(x)] for x in range(1,101)], 'yq':[['no']]*100, 'aux':[]}
    episode_to_file(os.path.join(mydir,'special','16.txt'),S)
    xs = [16, 8, 2, 64]
    S = {'xs':[[str(x)] for x in xs], 'ys':[['yes']]*len(xs),
     'xq':[[str(x)] for x in range(1,101)], 'yq':[['no']]*100, 'aux':[]}
    episode_to_file(os.path.join(mydir,'special','16_8_2_64.txt'),S)
    xs = [16, 23, 19, 20]
    S = {'xs':[[str(x)] for x in xs], 'ys':[['yes']]*len(xs),
     'xq':[[str(x)] for x in range(1,101)], 'yq':[['no']]*100, 'aux':[]}
    episode_to_file(os.path.join(mydir,'special','16_23_19_20.txt'),S)

def convert_nums_to_symbols(sample):
    sample['xs'] = [[str(x)] for x in sample['xs']]
    sample['xq'] = [[str(x)] for x in sample['xq']]
    sample['ys'] = [[y] for y in sample['ys']]
    sample['yq'] = [[y] for y in sample['yq']]
    return sample

def episode_to_file(fn_out,sample):

    assert not os.path.isfile(fn_out) # file must not already exist
    fid = open(fn_out,'w')

    ns = len(sample['xs'])
    fid.write('*SUPPORT*\n')    
    for s in range(ns):
        fid.write('IN: ')
        fid.write(' '.join(sample['xs'][s]))
        fid.write(' OUT: ')
        fid.write(' '.join(sample['ys'][s]))
        fid.write('\n')
    fid.write('\n')

    nq = len(sample['xq'])
    fid.write('*QUERY*\n')    
    for q in range(nq):
        fid.write('IN: ')
        fid.write(' '.join(sample['xq'][q]))
        fid.write(' OUT: ')
        fid.write(' '.join(sample['yq'][q]))
        fid.write('\n')
    fid.write('\n')

    fid.write('*HYPOTHESIS*\n')
    if 'h_str' in sample['aux']:
        h_str = str(sample['aux']['h_str'])
        fid.write(h_str)
    fid.close()


def num2str(x,mylen=6):
    # convert number x to string and add preceding zeros
    s = str(x)
    add = mylen -len(s)
    s = add*'0' + s
    return s    

if __name__ == "__main__":
    # generate_num_game_viz(30)
	# generate_num_game_dataset(10000,200)
    # generate_fol_concepts_uninform_dataset(200000,400)
    # generate_fol_concepts_viz(10, inc_complex=False)
    generate_fol_concepts_viz(10, inc_complex=False, mydir='data/temp')
