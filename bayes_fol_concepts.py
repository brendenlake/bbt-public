import os
import sys
from scipy.special import logsumexp
sys.path.insert(0,'/Users/Brenden/Libraries')
import numpy as np
import random
import pickle
from LOTlib3.Grammar import Grammar
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.FunctionHypothesis import FunctionHypothesis
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
from LOTlib3.Eval import primitive
from bayes_boolean_concepts import get_format_x, x_to_fdata, fdata_to_x_long, fdata_to_x_short, sample_set_objects
from copy import deepcopy

## first order logic concept learning with sets of examples, from Piantadosi et al. Psych Review paper
## Loosely based on 
# https://github.com/piantado/LOTlib/blob/master/LOTlib/Projects/SetFunctionLearning/Model/Grammar.py

symb_shapes = ['circle','triangle','rectangle']
symb_colors = ['blue','yellow','green']
symb_sizes = ['size_1','size_2','size_3']
symb_special = ['OBJ_SEP']

# hyperparameters
temperature = 4.75 # temperature of the prior (if used) 
alpha = 0.48 # controls noise model in the likelihood
beta_pos = -0.8 # memory parameters
beta_neg = -0.95

@primitive
def circle_(x):
    return x.shape=="circle"
@primitive
def triangle_(x):
    return x.shape=="triangle"
@primitive
def rectangle_(x):
    return x.shape=="rectangle"

@primitive
def blue_(x):
    return x.color=="blue"
@primitive
def yellow_(x):
    return x.color=="yellow"
@primitive
def green_(x):
    return x.color=="green"

@primitive
def size1_(x):
    return x.size=='size_1'
@primitive
def size2_(x):
    return x.size=='size_2'
@primitive
def size3_(x):
    return x.size=='size_3'

@primitive
# note that these functions work fine on strings
def size_geq_(x,y): return (x.size >= y.size)

@primitive
def size_gt_(x,y): return (x.size > y.size)

@primitive
def size_eq_(x,y): return (x.size == y.size)

@primitive
def color_eq_(x,y): return (x.color == y.color)

@primitive
def shape_eq_(x,y): return (x.shape == y.shape)

@primitive
def object_eq_(x,y): return (size_eq_(x,y) and color_eq_(x,y) and shape_eq_(x,y))

@primitive
def set_diff_custom_(S,elem): return set(S).difference(set([elem]))

@primitive
def exists_one_or_fewer_(F,S):
    return (sum(map(F,S)) <= 1)

@primitive
def exists_exactly_one_(F,S):
    return (sum(map(F,S)) == 1)

upweight = 5. # amount of upweight canaidate
grammar = Grammar()

# START
grammar.add_rule('START', 'True', None, 0.1)
grammar.add_rule('START', 'False', None, 0.4)
grammar.add_rule('START', '', ['BOOL'], 1.75*upweight) ## this will be set to UPWEIGHT in the uniform grammar

# SHAPE, COLOR, SIZE
for prd in ["circle_", "triangle_", "rectangle_", "yellow_", "green_", "blue_", "size1_", "size2_", "size3_"]:
    grammar.add_rule('PREDICATE', prd, ['OBJECT'], 1.0)

# SET
grammar.add_rule('SET', 'S', None, 1.6)
grammar.add_rule('SET', 'set_diff_custom_', ['S','x'], 1.35) ## 'Piantadosi's (non-Xes S)'

# RELN
#   Note, relations are not useful unless inside a lambda
grammar.add_rule('RELN', 'size_gt_', ['OBJECT','OBJECT'], 1.35)
grammar.add_rule('RELN', 'size_geq_', ['OBJECT','OBJECT'], 0.85) 
grammar.add_rule('RELN', 'size_eq_', ['OBJECT','OBJECT'], 0.1)
grammar.add_rule('RELN', 'shape_eq_', ['OBJECT','OBJECT'], 0.75)
grammar.add_rule('RELN', 'object_eq_', ['OBJECT','OBJECT'], 0.75)
grammar.add_rule('RELN', 'color_eq_', ['OBJECT','OBJECT'], 1.1)

# OBJECT
grammar.add_rule('OBJECT', 'x',  None,   1.9)
grammar.add_rule('OBJECT2BOOL', 'lambda', ['BOOL'], 1.0, bv_type='OBJECT', bv_p=0.5)

# BOOL
grammar.add_rule('BOOL', '', ['RELN'], 0.1*upweight)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 0.9)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 0.9)
grammar.add_rule('BOOL', 'implies_', ['BOOL', 'BOOL'], 0.9)
grammar.add_rule('BOOL', 'iff_', ['BOOL', 'BOOL'], 0.5)
grammar.add_rule('BOOL', 'forall_', ['OBJECT2BOOL','SET'], 0.1)
grammar.add_rule('BOOL', '', ['PREDICATE'], 1.6*upweight) ## this will be set to UPWEIGHT in the uniform grammar
grammar.add_rule('BOOL', 'exists_one_or_fewer_', ['OBJECT2BOOL','SET'], 0.1)
grammar.add_rule('BOOL', 'exists_', ['OBJECT2BOOL','SET'], 0.25)
grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.6)

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display='lambda S: lambda x: %s', **kwargs)

def generate_hypothesis():
    # sample from prior, without too many nodes
    while True:
        h = grammar.generate() 
        h = MyHypothesis(grammar=grammar,value=h)
        if h.value.count_nodes() < h.maxnodes:
            return h

def eval_hyp(x, h, x_parsed=None):
    # Evaluate the logical hypothesis on the data
    # Input
    # x : list of symbols, with different objects separated as OBJ_SEP
    # h : a hypothesis (as function)
    # x_parsed : (optional) pre-parsed result of x_to_fdata(x), avoids redundant parsing
    # Output
    # y : list of 'yes'/'no'
    if x_parsed is None:
        x_parsed = x_to_fdata(x)
    f = h(x_parsed)
    y_bool = [f(item) for item in x_parsed]
    assert all(isinstance(y, bool) for y in y_bool)
    return ['yes' if b else 'no' for b in y_bool]

def split_set_by_obj_sep(x):
    # Input
    #   x : object set (long or short format)
    #     If long format, split a set into a list of objects (each objcet as string) using the OBJ_SEP symbol as separator
    # Output
    #   list_of_obj : list of objects, each is a string of 3 symbols ['square green size_1', ... ]    
    if get_format_x(x) == 'long':
        x_str = ' '.join(x)
        return x_str.split(symb_special[0])
    return [item.replace('_', ' ').replace('size', 'size_') for item in x]

def set_grammar_weights_to_uniform(mygrammar_prev):
    # Produce uniformed version of the grammar
    #  if the weight is greater than upweights, set to upweight
    #  otherwise, set to 1.0
    mygrammar_prev = deepcopy(mygrammar_prev)
    for rule in mygrammar_prev.get_all_rules():
        if rule.p > upweight:
            rule.p = upweight
        else:
            rule.p = 1.0
    return mygrammar_prev

def grammar_add_exists_exactly_one(mygrammar_prev):
    # Add the exists_exactly_one rule to the grammar.
    # However, this was not in piantadosi's original grammar, so we add it just for viz/evaluation.
    mygrammar_prev = deepcopy(mygrammar_prev)
    mygrammar_prev.add_rule('BOOL', 'exists_exactly_one_', ['OBJECT2BOOL','SET'], 1.)
    return mygrammar_prev

def forward_predictive_model(nsamp_hyp, nsamp_support, nsamp_query):
    # Draw samples from the forward model
    #
    # INPUT
    #  nsamp_hyp : number of hypotheses to be sample from the prior
    #
    # RETURN
    #  list_H : [nsamp length python list] of sampled hypotheses (each a python list of numbers)
    samples = []
    for i in range(nsamp_hyp):

        # sample from p(h)
        h = generate_hypothesis()

        # sample objects x from independent distribution 
        xs = [sample_set_objects() for _ in range(nsamp_support)]
        xq = [sample_set_objects() for _ in range(nsamp_query)]

        # sample from p(y|h;x)
        ys = [eval_hyp(myset,h) for myset in xs]
        yq = [eval_hyp(myset,h) for myset in xq]
        S = {'xs':xs, 'ys':ys, 'xq':xq, 'yq':yq, 'aux':{'h_str':str(h)}}
        samples.append(S)
    return samples

def forward_model_special(nsamp_support, nsamp_query, h, nobj_min=1):
    # Draw samples from the forward model with a fixed hypothesis h
    #
    # INPUT
    #  nsamp_support : number of support samples to be drawn
    #  nsamp_query : number of query samples to be drawn
    #
    # RETURN
    #  S : dictionary with keys

    # sample objects x from independent distribution 
    xs = [sample_set_objects(nobj_min=nobj_min) for _ in range(nsamp_support)]
    xq = [sample_set_objects(nobj_min=nobj_min) for _ in range(nsamp_query)]

    # sample from p(y|h;x)
    ys = [eval_hyp(myset,h) for myset in xs]
    yq = [eval_hyp(myset,h) for myset in xq]    
    S = {'xs':xs, 'ys':ys, 'xq':xq, 'yq':yq, 'aux':{'h_str':str(h)}}
    return S

def forward_model_special_soft(nsamp_support, nsamp_query, h):
    # Draw samples from the forward model with a fixed hypothesis h.
    # Here, we add noise to the SUPPORT labels, but not the query labels
    #
    # INPUT
    #  nsamp_support : number of support samples to be drawn
    #  nsamp_query : number of query samples to be drawn
    #
    # RETURN
    #  S : dictionary with keys

    # sample objects x from independent distribution 
    xs = [sample_set_objects() for _ in range(nsamp_support)]
    xq = [sample_set_objects() for _ in range(nsamp_query)]

    # sample from p(y|h;x) with noisy likelihood
    ys = [eval_hyp(myset,h) for myset in xs]
    yq = [eval_hyp(myset,h) for myset in xq]

    # apply noise to support only
    for j in range(nsamp_support):
        for k in range(len(ys[j])):
            if flip(1-alpha):
                ys[j][k] = random.choice(['yes','no'])
    
    S = {'xs':xs, 'ys':ys, 'xq':xq, 'yq':yq, 'aux':{'h_str':str(h)}}
    return S

def cache_LOT_hypotheses(model_type):
    # Cache the LOT hypotheses from the dataset and estimate their log probabilities given their frequencies.
    #   store as fn_out, which is a dictionary with keys 'hypotheses' and 'logp'
    from datasets import get_dataset, readfile
    from collections import Counter
    if model_type == 'uniform':
        fn_out = 'data/set_concepts_uniform/set_concepts_uniform_hypotheses.pkl'
    elif model_type == 'temperature':
        fn_out = 'data/set_concepts_temperature/set_concepts_temperature_hypotheses.pkl'
    else:
        assert False, f'Invalid model_type: {model_type}'
    if os.path.exists(fn_out):        
        with open(fn_out, 'rb') as f:
            output = pickle.load(f)
            print('Loaded cached hypotheses from %s' % fn_out)
            return output['hypotheses'], output['logp']        
    if model_type == 'uniform':
        D_train, _ = get_dataset('set_concepts_uniform')
    elif model_type == 'temperature':
        D_train, _ = get_dataset('set_concepts_temperature')
    else:
        assert False, f'Invalid model_type: {model_type}'
    n = len(D_train)
    hyp_all = Counter()
    for i in range(n):
        if i % 5000 == 0: print(f'Processing {i}/{n}...')
        S = readfile(D_train.list_items[i])
        h_str = S['h_str']
        hyp_all[h_str] += 1
    print(f' number of unique hypotheses loaded: {len(hyp_all)}')
    list_hyp_all,list_logp = zip(*hyp_all.most_common())
    list_logp = np.array(list_logp, dtype=np.float32)
    list_logp = np.log(list_logp)
    list_logp = list_logp - logsumexp(list_logp)
    assert(np.isclose(np.sum(np.exp(list_logp)), 1.0))
    print(f' top 10 most common hypotheses: {list_hyp_all[:10]}'
          f' with log probabilities: {list_logp[:10]}')
    output = {'hypotheses':list_hyp_all, 'logp':list_logp}
    with open(fn_out, 'wb') as f:
        pickle.dump(output, f)
    return output['hypotheses'], output['logp']

def log_likelihood_hard(list_xs_parsed, list_ys, f_h):
    # Compute the log likelihood of multiple sets of objects and their labels, given a hypothesis (as function)
    # INPUT
    #  list_xs_parsed : list of sets of objects (each set is a list of LOTlib objects)
    #  list_ys : list of sets of labels to objects (each is a list of 'yes'/'no')
    #  f_h : a hypothesis (as a function that takes in a list of LOTlib objects and returns a function that takes in an object and returns bool)
    for idx, xs_parsed in enumerate(list_xs_parsed):
        if list_ys[idx] != eval_hyp(None, f_h, x_parsed=xs_parsed):
            return -np.inf
    return 0.0

def flip(p):
    # return True if with probability p, otherwise return False
    return random.random() < p

def log_likelihood_soft(list_xs_parsed, list_ys, f_h):
    # Compute the log likelihood of multiple sets of objects and their labels, given a hypothesis (as function).
    # Here we have a noise model
    # INPUT
    #  list_xs_parsed : list of sets of objects (each set is a list of LOTlib objects)
    #  list_ys : list of sets of labels to objects (each is a list of 'yes'/'no')
    #  f_h : a hypothesis (as a function that takes in a list of LOTlib objects and returns a function that takes in an object and returns bool)
    ll = 0.0    
    for idx, xs_parsed in enumerate(list_xs_parsed):
        ys_target = list_ys[idx]        
        ys_pred = eval_hyp(None, f_h, x_parsed=xs_parsed)
        for y_t, y_p in zip(ys_target, ys_pred):
            if y_t == y_p:
                ll += np.log(alpha + (1-alpha)*0.5)
            else:
                ll += np.log((1-alpha)*0.5)
    return ll

def log_likelihood_memory(list_xs_parsed, list_ys, f_h):
    # Compute the log likelihood of multiple sets of objects and their labels, given a hypothesis (as function).
    # Here we have a noise model and a memory model
    # INPUT
    #  list_xs_parsed : list of sets of objects (each set is a list of LOTlib objects)
    #   NOTE, these sets must be ordered in the sequence they were observed, with the earliest observation first and the latest observation 
    #  list_ys : list of sets of labels to objects (each is a list of 'yes'/'no')
    #  f_h : a hypothesis (as a function that takes in a list of LOTlib objects and returns a function that takes in an object and returns bool)
    ll = 0.0
    nsets = len(list_xs_parsed)
    for idx, xs_parsed in enumerate(list_xs_parsed):
        ys_target = list_ys[idx]        
        ys_pred = eval_hyp(None, f_h, x_parsed=xs_parsed)
        for y_t, y_p in zip(ys_target, ys_pred):
            if y_t == y_p:
                ll_dat = np.log(alpha + (1-alpha)*0.5)
            else:
                ll_dat = np.log((1-alpha)*0.5)
            beta = beta_pos if y_t == 'yes' else beta_neg
            i = idx + 1 # 1-indexed
            ll += (nsets - i + 1)**(-beta) * ll_dat   
    return ll

def log_posterior(list_xs_parsed, list_ys, eval_hyps, log_prior, ll_type='soft'):
    # Compute the log posterior probability of each hypothesis.
    # INPUT
    #  list_xs_parsed : list of sets of objects (each set is a list of LOTlib objects)
    #  list_ys : list of sets of labels to objects (each is a list of 'yes'/'no')
    #  eval_hyps : list of hypotheses (each is a function)
    #  log_prior : list of log probabilities of the hypotheses (same length as hypotheses)
    #
    # RETURN
    #  log_post : list of log posterior probabilities of the hypotheses (same length as hypotheses)
    nhyp = len(eval_hyps)
    ll = -np.inf * np.ones(nhyp, dtype=np.float32)
    for j, f_h in enumerate(eval_hyps):
        if ll_type == 'soft':
            ll[j] = log_likelihood_soft(list_xs_parsed, list_ys, f_h)
        elif ll_type == 'memory':
            ll[j] = log_likelihood_memory(list_xs_parsed, list_ys, f_h)
        elif ll_type == 'hard':
            ll[j] = log_likelihood_hard(list_xs_parsed, list_ys, f_h)
        else:
            raise ValueError(f'Invalid ll_type: {ll_type}')
    log_post = ll + log_prior
    log_post = log_post - logsumexp(log_post)
    return log_post

def bayesian_predictions(list_xq, list_xs, list_ys, list_hypotheses, log_prior, ll_type='soft'):
    # INPUT
    #  xq : [nq list] of new sets of objects we want to check the probability of membership for
    #     each set in xq is to be evaluated independently in the posterior predictive
    #  xs : [ns list] observed sets of objects (X), each of which is a list
    #  ys : [ns list] observed labels corresponding to each set in xs, each which is a list of 'yes'/'no'
    #  list_hypotheses : python list of hypotheses, each of which is a string
    #  log_prior : numpy vector [length nh] which is the log prior value for each hypothesis
    #
    # RETURN
    #  list_yq_pp : [nq list] of predicted probabilities of new sets in xq. each element is itself a numpy 1d array (NOTE: NOT IN LOG SPACE)
    #  best_hyp: the hypothesis with the highest posterior probability
    eval_hyps = [eval(h_str) for h_str in list_hypotheses]
    list_xs_parsed = [x_to_fdata(x) for x in list_xs]
    list_xq_parsed = [x_to_fdata(x) for x in list_xq]
    nhyp = len(list_hypotheses)

    log_post = log_posterior(list_xs_parsed, list_ys, eval_hyps, log_prior, ll_type=ll_type)
    widx = np.argmax(log_post)
    post = np.exp(log_post)

    list_yq_pp = []
    for xq_parsed in list_xq_parsed:
        m = len(xq_parsed)
        yq_pp = np.zeros((nhyp, m), dtype=np.float32)
        for j, f_h in enumerate(eval_hyps):
            yq_set_pp = eval_hyp(None, f_h, x_parsed=xq_parsed)
            assert all(y in ['yes', 'no'] for y in yq_set_pp)
            yq_pp[j] = [1.0 if y == 'yes' else 0.0 for y in yq_set_pp]
            # if ll_type =='hard':
            #     yq_pp[j] = [1.0 if y == 'yes' else 0.0 for y in yq_set_pp]
            # else:
            #     yq_pp[j] = [alpha + (1-alpha)*0.5 if y == 'yes' else (1-alpha)*0.5 for y in yq_set_pp]
        list_yq_pp.append(np.transpose(yq_pp) @ post) # shape (m,) = (nhyp, m).T @ (nhyp,) -> (m,)
    return list_yq_pp, (list_hypotheses[widx], log_post[widx].item()) 

if __name__ == '__main__':
    # grammar = set_grammar_weights_to_uniform(grammar) # uniform the grammar weights
    # print(grammar)
    # h = generate_hypothesis()
    # print(h)
    # xs = sample_set_objects()
    # print(xs)
    # ys = eval_hyp(xs,h)
    # print(ys)

    list_xs = [sample_set_objects(), sample_set_objects(), sample_set_objects()]
    list_ys = [eval_hyp(xs,lambda S: lambda x: True) for xs in list_xs]
    list_xq = [sample_set_objects(), sample_set_objects()]
    list_hypotheses, log_prior = cache_LOT_hypotheses(model_type='temperature')
    print(list_hypotheses[:10])
    print(len(list_hypotheses))
    yq_pp, best_hyp = bayesian_predictions(list_xq, list_xs, list_ys, list_hypotheses, log_prior, ll_type='soft')
    # print(list_xs)
    # print(list_ys)
    # print(list_xq)
    print(f'posterior predictive: {yq_pp}')
    print(f'best hypothesis: {best_hyp}')