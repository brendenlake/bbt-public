import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
import numpy as np
import pickle
from copy import deepcopy, copy
from sklearn import utils
from train_utils import seed_all, display_input_output, get_best_device
from collections import Counter
DEVICE = get_best_device()

DEBUG = False # compare outputs with legacy code

# Special symbols
SOS_token = "SOS" # start of sequence
EOS_token = "EOS" # end of sequence
PAD_token = "PAD" # padding symbol
IO_SEP = 'IO' # separator '->' between input/outputs in support examples
ITEM_SEP = SOS_token # separator '|' between support examples in input sequence

def get_dataset(episode_type):
    # Tenenbaum's nubmer game
    if episode_type == 'number_game':
        D_train = DataNumberGame('train')
        D_val = DataNumberGame('val')
    elif episode_type == 'number_game_special':
        D_train = []
        D_val = DataNumberGame('special')
    elif episode_type == 'number_game_human':
        D_train = DataNumberGameHuman('human_train')
        D_val = DataNumberGameHuman('human_val')
    elif episode_type == 'number_game_human_test':
        D_train = []
        D_val = DataNumberGameHuman('human_test')
    elif episode_type == 'number_game_probe':
        D_train = DataNumberGameProbe('train_probe', mydir='data/num_game_probe', nquery_inc=5)
        D_val = DataNumberGameProbe('val_probe', mydir='data/num_game_probe')
    elif episode_type == 'number_game_viz':
        D_train = []
        D_val = DataNumberGame('special', mydir='data/num_game_viz')

    # Piantadosi set (FOL) concepts
    elif episode_type == 'set_concepts_uniform':
        # 200K synthetic set concepts, generated from Piantadosi et al. grammar with flat (as possible) weights
        D_train = DataSetConceptsCompressed('train', inc_support_in_query=True, do_subsample_support=True, mydir='data/set_concepts_uniform')
        D_val = DataSetConceptsCompressed('val', inc_support_in_query=False, mydir='data/set_concepts_uniform')
    elif episode_type == 'set_concepts_human_ftune':
        # Piantadosi's training set that combines both boolean and FOL concepts and behavioral responses, split into a val set too
        D_train = DataSetConceptsCompressed('human_all_piantadosi_train', do_set_perm=True, inc_support_in_query=True, mydir='data/set_concepts')
        D_val = DataSetConceptsCompressed('human_all_piantadosi_val', inc_support_in_query=False, mydir='data/set_concepts')
    elif episode_type == 'set_concepts_human_test':
        # Piantadosi's test set of all concepts and behavioral responses
        D_train = []
        D_val = DataSetConceptsCompressed('human_all_piantadosi_test', inc_support_in_query=False, mydir='data/set_concepts')
    elif episode_type == 'set_concepts_human_test_disjoint':
        # Piantadosi's test set of all concepts, including concepts that all appeared in val set (not training)
        D_train = []
        D_val = DataSetConceptsCompressed('human_all_piantadosi_test_disjoint', inc_support_in_query=False, mydir='data/set_concepts')
    elif episode_type == 'set_concepts_gold_test':
        # Piantadosi's test set of all concepts and gold resopnses
        D_train = []
        D_val = DataSetConceptsCompressedGold('human_all_piantadosi_test', mydir='data/set_concepts')
    elif episode_type == 'set_concepts_viz':
        D_train = []
        D_val = DataSetConceptsCompressed('special', inc_support_in_query=False, mydir='data/set_concepts_viz')
    elif episode_type == 'set_concepts_viz_marker':
        D_train = []
        D_val = DataSetConceptsCompressed('special_marker', inc_support_in_query=False, mydir='data/set_concepts_viz')

    # Compositional rules
    elif episode_type == 'sys_gen': # equivalent to MLC (algebraic only) training, while varying the support set size
        D_train = DataAlg('train', mydir='data/sys_gen/data_algebraic', p_noise=0.01, min_ns=0)
        D_val = DataAlg('val', mydir='data/sys_gen/data_algebraic')
    elif episode_type=='sys_gen_gold': # For evaluating few-shot learning task with gold outputs
        D_train = []
        D_val = DataHumanFewShot('test_gold', inc_support_in_query=True, do_remap=True, mydir='data/sys_gen/data_human/few_shot_test')
    elif episode_type=='sys_gen_test_human': # For evaluating predictions of human behavior on few-shot learning task
        D_train = []
        D_val = DataHumanFewShot('test_behavior', data_mult=1, inc_support_in_query=False, do_remap=True, mydir='data/sys_gen/data_human/few_shot_test')
    elif episode_type=='sys_gen_test_human_mult10': # data multiplier gives us 10x more data
        D_train = []
        D_val = DataHumanFewShot('test_behavior', data_mult=10, inc_support_in_query=False, do_remap=True, mydir='data/sys_gen/data_human/few_shot_test')
    elif episode_type=='sys_gen_tune_human':
        D_train = DataHumanFewShot('train_behavior', data_mult=1, inc_support_in_query=False, do_remap=True, mydir='data/sys_gen/data_human/few_shot_tuning')
        D_val = DataHumanFewShot('val_behavior', data_mult=1, inc_support_in_query=False, do_remap=True, mydir='data/sys_gen/data_human/few_shot_tuning')
    elif episode_type=='sys_gen_alltune_human': # all the data is used for fine-tuning
        D_train = DataHumanFewShot('all', data_mult=1, inc_support_in_query=False, do_remap=True, mydir='data/sys_gen/data_human/few_shot_tuning')
        D_val = DataHumanFewShot('all', data_mult=1, inc_support_in_query=False, do_remap=True, mydir='data/sys_gen/data_human/few_shot_tuning')    
    elif episode_type == 'sys_gen_constrained': # used for selecting grammars for running humans on
        D_train = []
        D_val = DataAlg('val', data_mult=10, mydir='data/sys_gen/data_algebraic_constrained')        
    else:
        assert False # invalid episode types
    return D_train, D_val

class Lang:
    #  Class for converting tokens strings to token index, and vice versa.
    #   Use separate class for input and output languages
    #
    def __init__(self, symbols):
        # symbols : list of all possible symbols besides special tokens SOS, EOS, and PAD
        n = len(symbols)
        assert(SOS_token not in symbols)
        assert(EOS_token not in symbols)
        assert(PAD_token not in symbols)
        assert(PAD_token != SOS_token)
        self.symbols = symbols # list of non-special symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token, n+2: PAD_token}
        self.symbol2index = {SOS_token : n, EOS_token : n+1, PAD_token : n+2}
        for idx,s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)
        self.PAD_idx = self.symbol2index[PAD_token]        
        self.PAD_token = PAD_token
        self.EOS_idx = self.symbol2index[EOS_token]
        if ITEM_SEP in self.symbol2index:
            self.ITEM_idx = self.symbol2index[ITEM_SEP]
        assert(len(self.index2symbol)==len(self.symbol2index))

    def symbols_to_tensor(self, mylist, add_eos=True):
        # Convert a list of token strings to token index (adding a EOS token at end)
        # 
        # Input
        #  mylist  : list of m symbols as strings
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] token index for each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos: mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices) # keep on CPU since this occurs inside Dataset getitem..
        return output

    def tensor_to_symbols(self, v):
        # Convert tensor of token index to token strings, breaking where we get a EOS token.
        #   The EOS token is not included at the end in the result string list.
        # 
        # Input
        #  v : python list of m indices, or 1D tensor
        #   
        # Output
        #  mylist : list of symbols (excluding EOS)
        if torch.is_tensor(v):
            assert v.dim()==1
            v = v.tolist()
        assert isinstance(v, list)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

def bundle_bbt_episode(x_support,y_support,x_query,y_query,myhash,aux={}):
    # Bundle components for an episode suitable for optimizing BBT
    # 
    # Input
    #  x_support [length ns list of lists, or empty] : input sequences (each a python list of symbols)
    #  y_support [length ns list of lists] : output sequences (each a python list of symbols)
    #  x_query [length nq list of lists, or empty] : input sequences (each a python list of symbols)
    #  x_query [length nq list of lists] : output sequences (each a python list of symbols)
    #  myhash : unique string identifier for this episode (should be order invariant for examples)
    #  aux [dict] : any misc information that we want to pass along with the episode
    #
    # Output
    #  sample : dict that stores episode information
    S = helper_bundle_bbt_episode(x_support,y_support,x_query,y_query,myhash,aux=aux)
    return S

def helper_bundle_bbt_episode(x_support,y_support,x_query,y_query,myhash,aux={}):
    ns = len(y_support)
    # if x_support: # if we have x,y data pairs
    assert(x_query)
    xy_support = [ITEM_SEP]
    for j in range(ns):
        xy_support += x_support[j] + [IO_SEP] + y_support[j] + [ITEM_SEP]
    x_query_context = [item + xy_support for item in x_query]
    sample = {}
    sample['identifier'] = myhash # unique identifying string for this episode (order invariant)
    sample['xs'] = x_support # support 
    sample['ys'] = y_support
    sample['xq'] = x_query # query
    sample['yq'] = y_query
    sample['xq_context'] = x_query_context
    if aux: sample['aux'] = aux
    return sample


def make_bbt_batch(samples, langs):
    # Batch episodes into a series of padded input and target tensors
    # 
    # Input
    #  samples : list of dicts from bundle_bbt_episode
    #  langs : input and output version of Lang class
    return helper_make_bbt_batch(samples, langs)


def helper_make_bbt_batch(samples, langs):
    # Batch episodes into a series of padded input and target tensors
    # 
    # Input
    #  samples : list of dicts from bundle_bbt_episode
    #  langs : input and output version of Lang class
    assert isinstance(samples,list)
    m = len(samples)
    mybatch = {}
    mybatch['list_samples'] = samples
    mybatch['batch_size'] = m
    mybatch['xq_context'] = [] # list of source sequences (as lists) across all episodes
    mybatch['xq'] = []  # list of queries (as lists) across all episodes
    mybatch['yq'] = [] # list of query outputs (as lists) across all episodes
    mybatch['q_idx'] = [] # index of which episode each query belongs to
    mybatch['in_support'] = [] # bool list indicating whether each query is in its corresponding support set, or not
    for idx in range(m): # each episode
        sample = samples[idx]
        nq = len(sample['yq'])
        mybatch['xq_context'] += sample['xq_context']
        mybatch['xq'] += sample['xq']
        mybatch['yq'] += sample['yq']
        mybatch['q_idx'] += [idx*torch.ones(nq, dtype=torch.int)]
        mybatch['in_support'] += [x in sample['xs'] for x in sample['xq']]
    mybatch['q_idx'] = torch.cat(mybatch['q_idx'], dim=0)
    mybatch['xq_context_padded'],mybatch['xq_context_lengths'] = build_padded_tensor(mybatch['xq_context'], langs['input'])
    mybatch['yq_padded'],mybatch['yq_lengths'] = build_padded_tensor(mybatch['yq'], langs['output'])
    mybatch['yq_sos_padded'],mybatch['yq_sos_lengths'] = build_padded_tensor(mybatch['yq'],langs['output'],add_eos=False,add_sos=True)
        # has no SOS or EOS tokens added
    return mybatch


def set_batch_to_device(batch):
    # Make sure all padded tensors are on GPU if needed
    tensors_to_gpu = [k for k in batch.keys() if ('_padded' in k or 'ignore_mask' in k)]
    for k in tensors_to_gpu:
        batch[k] = batch[k].to(device=DEVICE)
    return batch

def build_padded_tensor(list_seq, lang, add_eos=True, add_sos=False, max_len=-1):
    # Transform list of python lists to a padded torch tensors
    # 
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of symbols)
    #  lang : language object for translation into indices
    #  pad_length : the maximum size of any sequence, including special characters
    #  add_eos : add end-of-sequence token at the end?
    #  add_sos : add start-of-sequence token at the beginning?
    #  max_len : default=-1, set length of padded sequences manually
    #
    # Output
    #  z_padded : LongTensor (n x max_len)
    #  z_lengths : python list of sequence lengths (list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = deepcopy(list_seq)
    if add_sos: 
        z_eos = [[SOS_token]+z for z in z_eos]
    if add_eos:
        z_eos = [z+[EOS_token] for z in z_eos]    
    z_lengths = [len(z) for z in z_eos]
    if max_len<0:
        max_len = max(z_lengths) # maximum length in this episode
    else:
        assert(max_len) >= max(z_lengths), "max_len variable should be longer than any data sequence"
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.symbols_to_tensor(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded, dim=0) # n x max_len
    return z_padded,z_lengths

def pad_seq(seq, max_length):
    # Pad token string sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols (as strings)
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += (max_length - len(seq)) * [PAD_token]
    return seq

def combine_input_output_symb(list_input_symb,list_output_symb):
    # Make new source vocabulary that combines list of input and output symbols.
    #  Include input/output and item separators, IO_SEP,ITEM_SEP.
    #  Exclude EOS_token,SOS_token,PAD_token, which will be added automatically by Lang constructor/
    #
    # Input
    #   list_input_symb : list of token symbols (strings)
    #   list_output_symb : list of token symbols (strings)
    # Output
    #   comb : combined list of tokens as strings
    assert(len(set(list_input_symb).intersection(set(list_output_symb)))==0), "in BBT, input and output vocabs should *not* share tokens"
    additional_symb = sorted(set([IO_SEP,ITEM_SEP])-set([EOS_token,SOS_token,PAD_token]))
    comb = sorted(set(list_input_symb + list_output_symb + additional_symb))
    return comb

def readfile(fn_in):
    # Read episode from text file
    #
    # Input
    #  fn_in : filename to read
    #
    # Output
    #   Parsed version of the file, with struct
    #   
    fid = open(os.path.join(fn_in),'r')
    lines = fid.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = [line for line in lines if line != '']
    idx_support = lines.index('*SUPPORT*')
    idx_query = lines.index('*QUERY*')
    if '*HYPOTHESIS*' in lines:
        idx_hypothesis = lines.index('*HYPOTHESIS*')
    else:
        idx_hypothesis = lines.index('*GRAMMAR*')
    x_support, y_support = parse_commands(lines[idx_support+1:idx_query])
    x_query, y_query = parse_commands(lines[idx_query+1:idx_hypothesis])
    h_str = lines[idx_hypothesis+1:]
    h_str = '\n'.join(h_str)
    fid.close()
    return {'xs':x_support, 'ys':y_support, 'xq':x_query, 'yq':y_query, 'h_str':h_str}

def parse_commands(lines):
    # Parse lines from input files into command sequence and output sequence
    #
    # Input
    #  lines: [list of strings], each of format "IN: a b c OUT: d e f""
    lines = [l.strip() for l in lines]
    lines = [l.lstrip('IN: ') for l in lines]
    D = [l.split(' OUT: ') for l in lines]
    x = [d[0].split(' ') for d in D]
    y = [d[1].split(' ') for d in D]
    return x, y

class DataNumberGame(Dataset):
    # Meta-training for the number game (synthetic data from Bayesian model)

    def __init__(self, mode, mydir='data/num_game'):        
        # Input
        #  mode: 'train', 'val', or 'special'
        #  mydir : directory where data is stored
        # assert mode in ['train','val','special','human_train','human_val']
        assert mode in ['train','val','special']
        self.mode = mode
        self.train = mode == 'train' or mode == 'human_train'
        self.special = mode == 'special'
        self.special_mult = 1 # number of times we should multiply the queries in special episode
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = [str(i) for i in range(1,101)]
        self.output_symbols = ['yes','no']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.min_ns = 1 # min number of support items in episode
        self.max_ns = 10 # max number of support items in episode

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        if self.special:
            S['xq'] = S['xq']*self.special_mult
            S['yq'] = S['yq']*self.special_mult
            myhash = S['h_str']
            aux = {'h_str': myhash}
            return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux=aux)
        if self.train:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        ns = random.randint(self.min_ns,self.max_ns)
        S['xs'] = S['xs'][:ns]
        S['ys'] = S['ys'][:ns]
        myhash = S['h_str']
        aux = {'h_str': myhash}
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)        
    

class DataNumberGameProbe(Dataset):
    # Probe with the exact posterior over hypotheses for the number game

    def __init__(self, mode, mydir='data/num_game_probe', nquery_inc=-1):        
        # Input
        #  mydir : directory where data is stored
        #  nquery_inc : number of query examples to include (-1 to include all)
        assert mode in ['train_probe','val_probe']
        self.mode = mode
        self.nquery_inc = nquery_inc
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = [str(i) for i in range(1,101)]
        self.output_symbols = ['yes','no']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        if self.nquery_inc > 0:
            S['xq'] = S['xq'][:self.nquery_inc]
            S['yq'] = S['yq'][:self.nquery_inc]
        mylines = S['h_str'].split('\n')
        mylines = [l for l in mylines if len(l)>0]
        h_str = mylines[0]
        myhash = h_str
        pp = mylines[1]
        post = [l.split(' ') for l in mylines[2:]]
        post_val = torch.tensor([float(l[0]) for l in post])
        post_label = [l[1] for l in post]
        aux = {'h_str': myhash, 'post_val':post_val, 'post_label':post_label, 'fn':self.list_items[idx]}
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

class DataNumberGameHuman(Dataset):
    # Meta-training for the number game (human behavior)

    def __init__(self, mode, mydir='data/num_game'):
        # Input
        #  mode: 'train', 'val', or 'special'
        #  mydir : directory where data is stored
        assert mode in ['human_train','human_val','human_test']
        self.mode = mode
        self.train = mode == 'human_train'
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = [str(i) for i in range(1,101)]
        self.output_symbols = ['yes','no']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.max_ns = 4 # max number of support items in episode

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        # do not change order for test items, to keep consistent with participant presentation
        S = readfile(self.list_items[idx])
        if self.train:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        myhash = S['h_str']
        aux = {'h_str': myhash}
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)
    
    def get_base_rate(self):
        # computer the ratio of yq responses that are 'yes vs. 'no'
        yq_all = []
        for idx in range(len(self.list_items)):
            S = readfile(self.list_items[idx])
            yq_all += S['yq']
        assert(all([len(item)==1 for item in yq_all])), "all output sequences are length 1"
        yq_all = np.array([item[0] for item in yq_all])
        n = float(len(yq_all))
        n_yes = np.sum(yq_all=='yes')
        n_no = np.sum(yq_all=='no')
        print('prob_yes',round(100.*n_yes/n,3)
              ,'; prob_no',round(100.*n_no/n,3),'; n=',n)
        assert(n_no+n_yes==n)

class DataSetConceptsCompressed(Dataset):
    # This version uses a more efficient tokenization of the set concepts
    def __init__(self, mode, mydir='data/set_concepts', inc_support_in_query=True, do_subsample_support=False, do_set_perm=False):        
        # Input
        #  mode: specific directory with episodes (['train','val','human_all_piantadosi_train','human_all_piantadosi_val',..])
        #  mydir : parent directory where data is stored
        #  inc_support_in_query : (True/false), provide support examples as additional queries
        #  do_subsample_support : (true/False), shuffle and sub-select the support examples
        #  do_set_perm : (true/False), shuffle the order of examples in each set, both for support and queries
        # assert mode in ['train','val','human_all_piantadosi_train','human_all_piantadosi_val',
                    # 'human_bool_piantadosi_test','human_all_piantadosi_test']
        self.mode = mode
        self.inc_support_in_query = inc_support_in_query
        self.do_subsample_support = do_subsample_support
        self.do_set_perm = do_set_perm
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        symb_obj = []
        for shape in ['circle','triangle','rectangle']:
            for color in ['blue','yellow','green']:
                for size in ['size1','size2','size3']:
                    symb_obj.append(f'{shape}_{color}_{size}')
        self.input_symbols = symb_obj
        self.output_symbols = ['yes','no']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.min_ns = 0 # min number of support items in episode
        self.max_ns = 25 # max number of support items in episode

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])

        # sub-sample the support examples
        total_ns_file = len(S['xs'])
        max_ns_spec = min(self.max_ns,total_ns_file)
        ns = max_ns_spec
        if self.do_subsample_support: # randomize order and split
            assert('train' in self.mode), "error: we only want to shuffle/randomize support during training"
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])            
            ns = random.randint(self.min_ns,max_ns_spec)
        if self.do_set_perm:
            S['xs'], S['ys'] = shuffle_order_within_set(S['xs'], S['ys'])
            S['xq'], S['yq'] = shuffle_order_within_set(S['xq'], S['yq'])
        S['xs'] = [set_concept_otoken_to_ctoken(' '.join(otoken)).split() for otoken in S['xs']]
        S['xq'] = [set_concept_otoken_to_ctoken(' '.join(otoken)).split() for otoken in S['xq']]
        S['xs'] = S['xs'][:ns]
        S['ys'] = S['ys'][:ns]
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        myhash = S['h_str']
        myhash = myhash + '\n' + 'fn="' + self.list_items[idx] + '"'
        aux = {'h_str': myhash}
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

class DataSetConceptsCompressedGold(DataSetConceptsCompressed):
    # Meta-test for the Piantadosi's set concepts.
    # Here, the labels are swapped to the *gold labels* rather than human behavior
    def __init__(self, mode, mydir='data/set_concepts'):
        assert(mode=='human_all_piantadosi_test')
        super().__init__(mode, mydir=mydir, inc_support_in_query=False, do_subsample_support=False, do_set_perm=False)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        total_ns_file = len(S['xs'])
        if total_ns_file > self.max_ns:
            assert False, "error: we should not have more than max_ns support examples in the gold set"            
        S['xs'] = [set_concept_otoken_to_ctoken(' '.join(otoken)).split() for otoken in S['xs']]
        S['xq'] = [set_concept_otoken_to_ctoken(' '.join(otoken)).split() for otoken in S['xq']]
        myhash = S['h_str']
        myhash = myhash + '\n' + 'fn="' + self.list_items[idx] + '"'
        aux = {'h_str': myhash}
        myhash = myhash.split('\n')
        mygold = myhash[1]
        assert('gold_labels=[' in mygold)
        mygold = mygold.replace('gold_labels=[','').rstrip(']')
        mygold = mygold.split(', ')
        mygold = [item.strip("'") for item in mygold]
        S['yq'] = [mygold for y in S['yq']]
        assert all([S['xq'][i]==S['xq'][0] for i in range(len(S['xq']))])
        assert all([S['yq'][i]==S['yq'][0] for i in range(len(S['yq']))])
        # S['xq'] = [S['xq'][0]]
        # S['yq'] = [S['yq'][0]]
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

def set_concept_otoken_to_ctoken(otoken):
    # Example
    #  Input:  rectangle blue size_3 OBJ_SEP rectangle green size_1 OBJ_SEP rectangle blue size_2 OBJ_SEP triangle green size_1
    #  Output: rectangle_blue_size3 rectangle_green_size1 rectangle_blue_size2 triangle_green_size1
    objects = otoken.split('OBJ_SEP')
    ctokens = []
    for obj in objects:
        parts = obj.strip().split()  # e.g. ['rectangle', 'blue', 'size_3']
        merged = '_'.join(p.replace('_', '') for p in parts)  # 'size_3' -> 'size3'
        ctokens.append(merged)
    return ' '.join(ctokens)

def set_concept_ctoken_to_otoken(ctoken):
    # Example
    #  Input: rectangle_blue_size3 rectangle_green_size1 rectangle_blue_size2 triangle_green_size1
    #  Output: rectangle blue size_3 OBJ_SEP rectangle green size_1 OBJ_SEP rectangle blue size_2 OBJ_SEP triangle green size_1
    objects = ctoken.strip().split()  # e.g. ['rectangle_blue_size3', ...]
    otokens = []
    for obj in objects:
        parts = obj.split('_')  # e.g. ['rectangle', 'blue', 'size3']
        parts = [p.replace('size', 'size_') for p in parts]  # 'size3' -> 'size_3'
        otokens.append(' '.join(parts))
    return ' OBJ_SEP '.join(otokens)

class DataSetConcepts(Dataset):
    # Meta-training for the Piantadosi's set concepts (synthetic data from Bayesian model).
    #
    # for training, sample number of support examples uniformly min_ns to max_ns
    # for validation, use all of the support examples up to max_ns
    def __init__(self, mode, mydir='data/set_concepts', inc_support_in_query=True, do_subsample_support=False, do_set_perm=False):        
        # Input
        #  mode: specific directory with episodes (['train','val','human_all_piantadosi_train','human_all_piantadosi_val',..])
        #  mydir : parent directory where data is stored
        #  inc_support_in_query : (True/false), provide support examples as additional queries
        #  do_subsample_support : (true/False), shuffle and sub-select the support examples
        #  do_set_perm : (true/False), shuffle the order of examples in each set, both for support and queries
        # assert mode in ['train','val','human_all_piantadosi_train','human_all_piantadosi_val',
                    # 'human_bool_piantadosi_test','human_all_piantadosi_test']
        self.mode = mode
        self.inc_support_in_query = inc_support_in_query
        self.do_subsample_support = do_subsample_support
        self.do_set_perm = do_set_perm
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        symb_shapes = ['circle','triangle','rectangle']
        symb_colors = ['blue','yellow','green']
        symb_sizes = ['size_1','size_2','size_3']
        symb_special = ['OBJ_SEP']
        self.input_symbols = symb_shapes+symb_colors+symb_sizes+symb_special
        self.output_symbols = ['yes','no']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.min_ns = 0 # min number of support items in episode
        self.max_ns = 25 # max number of support items in episode

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])

        # sub-sample the support examples
        total_ns_file = len(S['xs'])
        max_ns_spec = min(self.max_ns,total_ns_file)
        ns = max_ns_spec
        if self.do_subsample_support: # randomize order and split
            assert('train' in self.mode), "error: we only want to shuffle/randomize support during training"
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])            
            ns = random.randint(self.min_ns,max_ns_spec)
        if self.do_set_perm:
            S['xs'], S['ys'] = shuffle_order_within_set(S['xs'], S['ys'])
            S['xq'], S['yq'] = shuffle_order_within_set(S['xq'], S['yq'])
        S['xs'] = S['xs'][:ns]
        S['ys'] = S['ys'][:ns]
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        myhash = S['h_str']
        myhash = myhash + '\n' + 'fn="' + self.list_items[idx] + '"'
        aux = {'h_str': myhash}
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

    def count_unique_hyp(self):
        list_h = [readfile(fn)['h_str'] for fn in self.list_items]
        unique_h = sorted(set(list_h))
        print('number of unique hypotheses:', len(unique_h), 'out of', len(list_h))
        freq_counter = Counter(list_h)
        most_common = freq_counter.most_common(200)
        print('\n most frequent hypotheses:')
        for hyp, freq in most_common:
            print(f'{hyp}: {freq}')

    def get_base_rate(self):
        # computer the ratio of yq responses that are 'yes vs. 'no'
        yq_all = []
        for idx in range(len(self.list_items)):
            S = readfile(self.list_items[idx])
            for yq_for_set in S['yq']: # yq_for_set is a list of labels for each query in this set                
                yq_all += yq_for_set
        yq_all = np.array(yq_all)
        n = float(len(yq_all))
        n_yes = np.sum(yq_all=='yes')
        n_no = np.sum(yq_all=='no')
        print('prob_yes',round(100.*n_yes/n,3)
              ,'; prob_no',round(100.*n_no/n,3),'; n=',n)
        assert(n_no+n_yes==n)

def shuffle_order_within_set(x_pair_list,y_pair_list):
    # Input:
    #        x_pair_list : Like xs or xq, such that it is a list of lists. Each sub-list is a Piantadosi set
    #        y_pair_list : 
    # Output:
    #       same thing, now with the objects (and labels) in each set shuffled
    n = len(x_pair_list)
    for i in range(n):
        if len(x_pair_list[i])>0:
            x_str = ' '.join(x_pair_list[i])
            x_as_objs = x_str.split('OBJ_SEP')
            x_as_objs, y_pair_list[i] = utils.shuffle(x_as_objs, y_pair_list[i])
            x_pair_list[i] = (' OBJ_SEP '.join(x_as_objs)).split()
    return x_pair_list, y_pair_list

class DataSetConceptsGold(DataSetConcepts):
    # Meta-test for the Piantadosi's set concepts.
    # Here, the labels are swapped to the *gold labels* rather than human behavior
    def __init__(self, mode, mydir='data/set_concepts'):
        assert(mode=='human_all_piantadosi_test')
        super().__init__(mode, mydir=mydir, inc_support_in_query=False, do_subsample_support=False)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])

        # sub-sample the support examples
        total_ns_file = len(S['xs'])
        if total_ns_file > self.max_ns:
            assert False, "error: we should not have more than max_ns support examples in the gold set"
        max_ns_spec = min(self.max_ns,total_ns_file)
        ns = max_ns_spec
        S['xs'] = S['xs'][:ns]
        S['ys'] = S['ys'][:ns]
        myhash = S['h_str']
        myhash = myhash + '\n' + 'fn="' + self.list_items[idx] + '"'
        aux = {'h_str': myhash}
        myhash = myhash.split('\n')
        mygold = myhash[1]
        assert('gold_labels=[' in mygold)
        mygold = mygold.lstrip('gold_labels=[').rstrip(']')
        mygold = mygold.split(', ')
        mygold = [item.strip("'") for item in mygold]
        S['yq'] = [mygold for y in S['yq']]
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

class DataSetAlienClassif(Dataset):
    # for training, sample number of support examples uniformly min_ns to max_ns
    # for validation, use all of the support examples up to max_ns
    def __init__(self, mode, mydir='data/alien_figures_classif', inc_support_in_query=True, do_subsample_support=False):
        # Input
        #  mode: specific directory with episodes ()
        #  mydir : parent directory where data is stored
        #  inc_support_in_query : (True/false), provide support examples as additional queries
        #  do_subsample_support : (true/False), shuffle and sub-select the support examples
        assert mode in ['train','val']
        self.mode = mode
        self.inc_support_in_query = inc_support_in_query
        self.do_subsample_support = do_subsample_support
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        symb_shapes = ['A','B','C','D']
        symb_attachments = [str(i) for i in range(4)]
        symb_rot = ['+0','+90','+180','+270']
        self.input_symbols = symb_shapes+symb_attachments+symb_rot
        self.output_symbols = ['yes','no']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.min_ns = 1 # min number of support items in episode
        self.max_ns = 6 # max number of support items in episode

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])

        # sub-sample the support examples
        total_ns_file = len(S['xs'])
        max_ns_spec = min(self.max_ns,total_ns_file)
        ns = max_ns_spec
        if self.do_subsample_support: # randomize order and split
            assert('train' in self.mode), "error: we only want to shuffle/randomize support during training"
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
            ns = random.randint(self.min_ns,max_ns_spec)
        S['xs'] = S['xs'][:ns]
        S['ys'] = S['ys'][:ns]
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        myhash = S['h_str']
        aux = {'h_str': myhash}
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

class DataSetAlienGenerate(Dataset):
    # for training, sample number of support examples uniformly min_ns to max_ns
    # for validation, use all of the support examples up to max_ns
    def __init__(self, mode, mydir='data/alien_figures_generate', do_subsample_support=False):
        # Input
        #  mode: specific directory with episodes ()
        #  mydir : parent directory where data is stored
        #  do_subsample_support : (true/False), shuffle and sub-select the support examples
        assert mode in ['train','val']
        self.mode = mode
        self.do_subsample_support = do_subsample_support
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        symb_shapes = ['A','B','C','D']
        symb_attachments = [str(i) for i in range(4)]
        symb_rot = ['+0','+90','+180','+270']
        self.input_symbols = ['generate']
        self.output_symbols = symb_shapes+symb_attachments+symb_rot
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.min_ns = 1 # min number of support items in episode
        self.max_ns = 6 # max number of support items in episode

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        total_ns_file = len(S['xs'])
        max_ns_spec = min(self.max_ns,total_ns_file)
        ns = max_ns_spec
        if self.do_subsample_support: # randomize order and split
            assert('train' in self.mode), "error: we only want to shuffle/randomize support during training"
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
            ns = random.randint(self.min_ns,max_ns_spec)
        S['xs'] = S['xs'][:ns]
        S['ys'] = S['ys'][:ns]
        myhash = S['h_str']
        aux = {'h_str': myhash}
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

class DataAlg(Dataset):
    # Meta-training for compositional rule learning (synthetic data from Bayesian model)

    def __init__(self, mode, mydir, p_noise=0., inc_support_in_query=True, min_ns=0, data_mult=1):
        # Each episode has different latent (algebraic) grammar. 
        #  The number of support items picked uniformly from min_ns...max_ns
        #
        # Input
        #  mode: 'train' or 'val'
        #  mydir : directory where data is stored
        #  p_noise : for a given symbol emission, probability that it will be from uniform distribution
        #  inc_support_in_query : default=True. Boolean. Should support items also be query items?
        #  min_ns : min number of support items in episode
        assert mode in ['train','val']
        self.mode = mode
        self.train = mode == 'train'
        self.p_noise = p_noise
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = ['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer']
        self.output_symbols = ['RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.min_ns = min_ns # min number of support items in episode
        self.max_ns = 14 # max number of support items in episode
        self.data_mult = data_mult
        self.inc_support_in_query = inc_support_in_query
        self.data_mult = data_mult

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        max_ns = len(S['xs'])
        if self.train:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
            ns = random.randint(self.min_ns,max_ns)
            S['xs'] = S['xs'][:ns]
            S['ys'] = S['ys'][:ns]
        if self.data_mult>1:
            S['xq'] = S['xq']*self.data_mult
            S['yq'] = S['yq']*self.data_mult
        if self.p_noise > 0: # emission noise
            for i in range(len(S['yq'])):
                S['yq'][i] = add_response_noise(S['yq'][i],self.p_noise,self.langs)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        myhash = make_hashable_grammar(S['h_str'])
        aux = {'h_str':S['h_str']}        
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

def flip(p_head=0.5):
    # return True with probability p_head
    return random.random() < p_head

def make_hashable_grammar(G):
    # Create unique identifier for episodes defined by a sys_gen grammar.
    #  Separate and sort rules in string format.
    #
    # Input
    #   G : string of elements separated by \n specifying the structure of an episode 
    G_str = str(G).split('\n')
    G_str = [s.strip() for s in G_str]
    G_str = [s for s in G_str if s != ''] # remove empty strings
    G_str.sort()
    out = '\n'.join(G_str)
    return out.strip()

def add_response_noise(yq_item, p_noise, langs):
    # Flip coin with weight p_noise of heads. If heads, replace output symbol with an arbitrary one.
    #   This symbol could be EOS token, in which case, we end the example.
    #
    # Input
    #  yq_item : list of output symbols for the response to a single command (excluding EOS)
    #  p_noise : probability of lapse (uniform draw) for a particular emission
    #
    assert(isinstance(yq_item,list))
    assert(isinstance(yq_item[0],str))
    choice_set_wo_eos = langs['output'].symbols
    choice_set = langs['output'].symbols + [EOS_token]
    yq_item = deepcopy(yq_item)
    for i in range(len(yq_item)): # for each token
        if flip(p_noise):
            new_token = random.choice(choice_set)
            if i==0: new_token = random.choice(choice_set_wo_eos)
            if new_token == EOS_token:
                return yq_item[:i]
            yq_item[i] = new_token
    return yq_item

class DataHumanOpenEnded(Dataset):
    # Human data for open-ended task in compositional rule learning
    
    def __init__(self, mode, mydir, inc_support_in_query=False):
        # Input
        #  mode : 'train' 'val' or 'all' data
        #  inc_support_in_query : (default=False) include all support examples as queries?
        #        If True, it messes up the log-likelihood calculations.
        #  mydir : directory where data is stored
        assert mode in ['train','val','all']
        self.mode = mode
        self.train = (mode == 'train') or (mode == 'all')
        self.val = mode == 'val'
        self.inc_support_in_query = inc_support_in_query
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = ['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer']
        self.output_symbols = ['RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.max_ns = 6 # max number of support items in episode

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        assert(len(S['xs'])==0) # in the text files, all items should be listed as queries
        if self.val: # for validation, we want to keep the random seed fixed
            np_state = np.random.get_state()
            np.random.seed(idx)
            r_state = random.getstate()
            random.seed(idx)
        S['xq'], S['yq'] = utils.shuffle(S['xq'], S['yq']) # uses np.random seed
        S = assign_random_map(S, self.input_symbols, self.output_symbols)
        S = self.__random_query_split_uniform(S)
        if self.val:
            np.random.set_state(np_state)
            random.setstate(r_state)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']  
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux={})

    def __random_query_split_uniform(self, S):
        # Randomly split the examples between query and support, uniformly over support size
        x = S['xs'] + S['xq']
        y = S['ys'] + S['yq']
        x, y = utils.shuffle(x,y)
        n = len(x)
        ns = random.randint(0,n-1)
        S['xs'] = x[:ns]
        S['ys'] = y[:ns]
        S['xq'] = x[ns:]
        S['yq'] = y[ns:]
        return S
    
def assign_random_map(S, new_input_symbols, new_output_symbols):
    # Each input and output symbol in the episode file are re-assigned based on 
    #   the standard set of input/output symbols
    #
    # Input
    #  S : dict with 'xs', 'ys', 'xq', 'yq' input/output examples in old spaces
    #  new_input_symbols : list of new input symbols we want to use
    #  new_output_symbols : list of new output symbols we want to use
    #
    # Output
    #   S : updated dict with remapped input/output patterns
    S_raw = S
    S = deepcopy(S)
    ns = len(S['xs'])
    nq = len(S['xq'])
    list_x = S['xs'] + S['xq']
    list_x = sum(list_x, [])
    unique_in_symb = sorted(set(list_x)) # all original input symbols in episode
    
    # collect current output symbols    
    list_y = S['ys'] + S['yq']
    list_y = sum(list_y, [])
    unique_out_symb = sorted(set(list_y)) # all original output symbols in episode

    # construct mappings
    lang_in_symb = deepcopy(new_input_symbols)
    lang_out_symb = deepcopy(new_output_symbols)
    random.shuffle(lang_in_symb)
    random.shuffle(lang_out_symb)
    lang_in_symb_trunc = lang_in_symb[:len(unique_in_symb)] # truncate if all symbols aren't needed
    lang_out_symb_trunc = lang_out_symb[:len(unique_out_symb)] # truncate if all symbols aren't needed
    map_input = lambda command: list_remap(command,unique_in_symb,lang_in_symb_trunc)
    map_output = lambda actions: list_remap(actions,unique_out_symb,lang_out_symb_trunc)
    
    # functions to invert mappings (emissions not used in query/support will be mapped to 'undefined_action')
    uo_lhs = lang_out_symb
    uo_rhs = unique_out_symb + ['undefined_action']*(len(uo_lhs)-len(unique_out_symb))
    unmap_input = lambda command: list_remap(command,lang_in_symb_trunc,unique_in_symb)
    unmap_output = lambda actions: list_remap(actions,uo_lhs,uo_rhs)

    S['aux'] = {}
    S['aux']['unmap_input'] = unmap_input
    S['aux']['unmap_output'] = unmap_output
    S['xs'] = list(map(map_input,S['xs']))
    S['xq'] = list(map(map_input,S['xq']))
    S['ys'] = list(map(map_output,S['ys']))
    S['yq'] = list(map(map_output,S['yq']))
    if DEBUG:
        S['aux']['list_input_mapping'] = list(zip(unique_in_symb,lang_in_symb_trunc))
        S['aux']['list_output_mapping'] = list(zip(unique_out_symb,lang_out_symb_trunc))
        S['aux']['S_raw'] = S_raw
    return S

def list_remap(list_old,list_source,list_target):
    #  For a given list, replace each token in "source" with the corresponding token in "target"
    # 
    # Input
    #  list_old : list of tokens where we will check each for a remap
    #  list_source : length k list of tokens to be replaced
    #  list_target : length k list of tokens that will replace the source tokens
    assert(len(list_source)==len(list_target))
    mydict = dict(zip(list_source,list_target))
    list_new = deepcopy(list_old)
    for i in range(len(list_new)):
        if list_new[i] in mydict:
            list_new[i] = mydict[list_new[i]]
    return list_new

class DataHumanFewShot(Dataset):
    # For evaluating predictions on human compositional rule learning behavior
    
    def __init__(self, mode, inc_support_in_query=False, do_remap=True, data_mult=1, fix_seed=-1, mydir='data/sys_gen/data_human/few_shot'):
        # Input
        #   mode : 'gold' for ground-truth responses for each command
        #          'behavior' for actual human responses
        #  inc_support_in_query : (default=False) include all support examples as queries? If True, it messes up the log-likelihood calculations.
        #  do_remap : [default=True] Randomly remap the input and output symbols to standard pseudowords and colors 
        #  data_mult : [default=1] replicate the queries this many times (useful if we want multiple samples from models)
        #  feed_seed : [default=-1] if >=0, fix the random seed for the dataset
        assert mode in ['test_gold','test_behavior','train_behavior','val_behavior','all']
        self.placeholder_length = 100 # placeholder number of episodes (if 'gold' mode)
        self.mode = mode
        self.is_gold = mode == 'test_gold'
        self.randomize_order = True
        self.do_remap = do_remap
        self.data_mult = data_mult
        self.fix_seed = fix_seed 
        self.inc_support_in_query = inc_support_in_query
        subdir = self.mode
        self.mydir_items = os.path.join(mydir,subdir)    
        self.fn_gold = os.path.join(mydir,'test_gold','mini_scan.txt') # gold filename
        self.list_items = glob.glob(self.mydir_items+'/*.txt') # all episode files
        if do_remap:
            self.input_symbols = ['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer']
            self.output_symbols = ['RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK']
        else:
            self.input_symbols = ['1','2','3','DAX','thrice','surround','after']
            self.output_symbols = ['1','2','3','DAX']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.max_ns = 14

    def __len__(self):
        if self.is_gold:
            return self.placeholder_length
        else:
            return len(self.list_items)

    def __getitem__(self, idx):
        if self.is_gold: # gold output strings
            S = readfile(self.fn_gold)
        else: # behavioral data
            S = readfile(self.list_items[idx])
        if self.fix_seed>0:
            np_state = np.random.get_state()
            np.random.seed(idx)
            r_state = random.getstate()
            random.seed(idx)
        if self.randomize_order:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        S['aux'] = {}
        if self.do_remap: S = assign_random_map(S, self.input_symbols, self.output_symbols)
        if self.fix_seed>0:
            np.random.set_state(np_state)
            random.setstate(r_state)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        if self.data_mult>1:
            S['xq'] = S['xq']*self.data_mult
            S['yq'] = S['yq']*self.data_mult
        return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux=S['aux'])
    
    def get_raw_item(self, idx):
        # Get raw-text version of an episode
        assert not self.is_gold, "only applicable for behavioral data"
        S = readfile(self.list_items[idx])
        return S
    
# class DataHumanFewShotLapseExamples(DataHumanFewShot):
#     # For evaluating predictions on human compositional rule learning behavior.
#     #  In this version, support examples can be dropped at random.
    
#     def __init__(self, mode, p_drop_example=float('nan'), inc_support_in_query=False, do_remap=True, data_mult=1, fix_seed=-1, mydir='data/sys_gen/data_human/few_shot'):
#         # Input
#         #   mode : 'gold' for ground-truth responses for each command
#         #          'behavior' for actual human responses
#         #  inc_support_in_query : (default=False) include all support examples as queries? If True, it messes up the log-likelihood calculations.
#         #  do_remap : [default=True] Randomly remap the input and output symbols to standard pseudowords and colors 
#         #  data_mult : [default=1] replicate the queries this many times (useful if we want multiple samples from models)
#         #  feed_seed : [default=-1] if >=0, fix the random seed for the dataset
#         assert mode in ['behavior']
#         super().__init__(mode, inc_support_in_query=inc_support_in_query, do_remap=do_remap, data_mult=data_mult, fix_seed=fix_seed, mydir=mydir)
#         self.p_drop_example = p_drop_example
#         assert(self.randomize_order)
#         assert(not self.inc_support_in_query)
#         assert(self.do_remap)
#         assert(self.data_mult==1)
#         assert(fix_seed==-1)
#         # print('** Constructing Dataset with support drop-out (p_drop_example=',self.p_drop_example,') **')

#     def set_p_drop(self, p):
#         # Set the probability of dropping an example
#         self.p_drop_example = p
#         print('** Setting p_drop_example=',self.p_drop_example,'**')

#     def __getitem__(self, idx):
#         assert(not np.isnan(self.p_drop_example)), "p_drop_example should be set before calling __getitem__"
#         S = readfile(self.list_items[idx])
#         S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
#         S['aux'] = {}
#         new_xs = []
#         new_ys = []
#         for x, y in zip(S['xs'], S['ys']):
#             if flip(self.p_drop_example):
#                 continue # drop this example
#             else:
#                 new_xs.append(x)
#                 new_ys.append(y)
#         S['xs'], S['ys'] = new_xs, new_ys
#         S = assign_random_map(S, self.input_symbols, self.output_symbols)
#         return bundle_bbt_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux={})

if __name__ == "__main__":
    
    # D_train, D_val = get_dataset('set_concepts')
    # list_h = set()
    # for S in D_val:
    #     list_h.add(S['aux']['h_str'].split('\n')[0])  # take only the first line of the hypothesis string
    # list_h = list(list_h)
    # list_h.sort(key=len)
    # for h in list_h:
    #     print(h)

    seed_all(3)
    # D_train, D_val = get_dataset('number_game')
    # D_train, D_val = get_dataset('set_concepts_uniform')
    D_train, D_val = get_dataset('sys_gen')
    print('number of training episodes:', len(D_train))
    print('number of validation episodes:', len(D_val))
    sample = D_val[2]
    print("")
    print('*Study examples*')
    display_input_output(sample['xs'],sample['ys'],sample['ys'])
    print("")
    print('*Query examples*')
    display_input_output(sample['xq'],sample['yq'],sample['yq'])
    # print("")
    # print('*Context examples*')
    # display_input_output(sample['xq_context'],sample['yq'],sample['yq'])
    print('*Hypothesis*')
    if 'h_str' in sample['aux']:
        print(sample['aux']['h_str'])