import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from train_utils import seed_all, get_best_device
from datasets import readfile
import bayes_num_game as bng
DEVICE = get_best_device()

# Fit a sparse posterior predictive approximation to the posterior in a Bayesian number game
def inv_sigmoid(x): return torch.log(x / (1 - x))

class SparsePPApprox_NumGame(nn.Module):
    # Approximate a posterior predictive distribution in number game task with a sparse set of hypotheses and arbitrary weights
    def __init__(self):
        super().__init__()
        self.H_all,_,self.H_names = bng.generate_hypotheses(bng.mylambda_default)
        self.H_all = torch.from_numpy(np.array(self.H_all)).to(torch.float32)
        self.H_all = self.H_all.to(device=DEVICE) # [nhyp x 101]
        nhyp = self.H_all.shape[0] # number of hypotheses
        self.h_weight = nn.Parameter(torch.ones(nhyp), requires_grad=True) # [nhyp]
        self.lapse_rate_weight = nn.Parameter(inv_sigmoid(torch.tensor(.1)),requires_grad=True)
        # self.b_rate_weight = nn.Parameter(inv_sigmoid(torch.tensor(.2)),requires_grad=True)
        self.b_rate_weight = nn.Parameter(inv_sigmoid(torch.tensor(.5)),requires_grad=False)
        self.reg_strength = float('nan')
    
    @property
    def lapse_rate(self): # scalar for lapse rate
        return torch.sigmoid(self.lapse_rate_weight)
    
    @property
    def b_rate(self): # scalar for base rate in lapse model
        return torch.sigmoid(self.b_rate_weight)

    @property
    def w(self): # normalized posterior weight [nhyp]
        return torch.softmax(self.h_weight,dim=0) 

    def _pp(self):
        # Posterior predictive distribution, with free parameters w
        # representing the problem-specific posterior
        #
        # Output
        #   probability of each new number belong to the set (NOT LOG)
        return torch.matmul(torch.transpose(self.H_all,0,1), self.w) # [101 x nhyp] x [nhyp] = [101]
    
    def forward(self):
        # Compute the posteiror predictive probability of each number in the set
        #
        # Output
        #  out : tensor [101 x 2], probability of respond 'yes' (col 0) or 'no' (col 1) to each query (row)
        #    NOT log prob
        pp = self._pp() # [101]
        pp_smooth = (1-self.lapse_rate)*pp + self.lapse_rate*self.b_rate # [101]
        out = torch.stack( (pp_smooth, 1-pp_smooth), dim=1) # [101 x 2]
        return out

def flat(yq):
    # Flatten a list of lists
    return [item for sublist in yq for item in sublist]

class SparsePPApprox_SetConcepts(SparsePPApprox_NumGame):
    # Approximate a posterior predictive distribution with a sparse set of hypotheses and arbitrary weights
    def __init__(self, xq, list_H):
        # xq : a list of queries, each of which is a set of objects
        # list_H : list of hypotheses to include in the approximation. these are python functions that can be evaluated on the queries xq to produce predictions.
        super().__init__()
        from bayes_fol_concepts import eval_hyp, split_set_by_obj_sep
        self.objects_flat = flat([[s + '(set'+str(idx)+')' for s in split_set_by_obj_sep(myset)] for idx,myset in enumerate(xq)])
        self.H_all = []
        self.H_names = []
        sel_trivial_hyp = []
        for i,h in enumerate(list_H):
            h_str = str(h)
            h_pred = flat([eval_hyp(myset,h) for myset in xq])
            assert all(y in ['yes', 'no'] for y in h_pred), "Each element must be 'yes' or 'no'"
            h_pred = [1.0 if y=='yes' else 0.0 for y in h_pred] # convert to 1/0
            h_pred = torch.tensor(h_pred, dtype=torch.float32)
            if (torch.all(h_pred == 1.0) or torch.all(h_pred == 0.0)) and ('False' not in h_str) and ('True' not in h_str):
                sel_trivial_hyp.append(True)
            else:
                sel_trivial_hyp.append(False)
                self.H_all.append(h_pred)
                self.H_names.append(str(h))
        self.H_all = torch.stack(self.H_all, dim=0).to(device=DEVICE) # [nhyp x nobj]
        self.h_weight = nn.Parameter(torch.ones(self.H_all.shape[0]), requires_grad=True) # [nhyp]
        self.list_H = [h for i,h in enumerate(list_H) if not sel_trivial_hyp[i]] # [nhyp]
        # print('number of non-trivial hypotheses', self.H_all.shape[0], 'out of', len(sel_trivial_hyp))
        self._prune_identical_hypotheses()
        self.b_rate_weight.requires_grad = False 
        self.b_rate_weight.data = inv_sigmoid(torch.tensor(0.5))
        # self.lapse_rate_weight,  self.reg_strength = float('nan') ## these are other parameters that are inherited

    def _prune_identical_hypotheses(self):
        # Remove duplicate hypotheses that make identical predictions on the queries.
        # When two rows of H_all are identical, keep the one with the shorter name.
        seen = {}  # maps row hash (bytes) -> index of best (shortest-name) hypothesis
        for i in range(self.H_all.shape[0]):
            key = self.H_all[i].cpu().numpy().tobytes()
            if key in seen:
                if complexity_num_func(self.H_names[i]) < complexity_num_func(self.H_names[seen[key]]):
                    seen[key] = i
            else:
                seen[key] = i
        keep_idx = sorted(seen.values())
        self.H_all = self.H_all[keep_idx]
        self.H_names = [self.H_names[i] for i in keep_idx]
        self.list_H = [self.list_H[i] for i in keep_idx]
        self.h_weight = nn.Parameter(torch.ones(self.H_all.shape[0]), requires_grad=True)
        # print('pruned %d identical hypotheses, kept %d unique hypotheses' % (nhyp - len(keep_idx), len(keep_idx)))

    def _pp(self):
        # Posterior predictive distribution, with free parameters w
        # representing the problem-specific posterior
        #
        # Output
        #   probability of each new number belong to the set (NOT LOG)
        return torch.matmul(torch.transpose(self.H_all,0,1), self.w) # [nobj x nhyp] x [nhyp] = [nobj]
    
def complexity_num_func(h_str):
    return h_str.count('_(') # the number of functions    

def loss_mean_kl(pp_approx,pp_target):
    # Input
    #  pp_approx : [nn x 2], model-produced probabilities, from SparsePPApprox or SparsePPApprox_SetConcepts forward 
    #  pp_target : [nn x 2], target probabilities
    #       neither is in log space
    # Output
    #  loss : scalar, mean KL divergence
    return F.kl_div(torch.log(pp_approx), pp_target, reduction='batchmean') # sum over columns, then mean over rows

def w_entropy(sspa):
    # Input: w is a probability distribution over hypotheses, not in log space
    # Return: the entropy of the model's prior distribution, used as a regularizer
    w = sspa.w
    non_zero_w = w[w > 0]
    return -torch.sum(non_zero_w * torch.log(non_zero_w))
  
def fit_sparse_pp_approx_ngame(pp_target, x_eval, nsteps=20, step_tol=1e-6, reg_strength =  5*1e-3, verbose=True):
    # Fit the sparse posterior predictive approximation for number game, minimizing KL and entropy
    # Input
    #   pp_target : [tensor 101 x 2], target probabilities for posterior predictive (col 0 is yes, col 1 is no)
    #   x_eval : tensor [list of int], numbers to evaluate
    #   reg_strength : strength of regularization
    # Output
    #   w : [tensor nhyp], the weight of each hypothesis
    #   pp_approx : [tensor 101 x 2], the posterior predictive probabilities of sparse approx
    #   sorted_w : [tensor nhyp], sorted weight of each hypothesis (descending)
    #   sorted_names : [list of str], sorted names of each hypothesis
    def closure():
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        pp_approx = sppa() # [101 x 2], log-probs
        pp_approx = pp_approx[x_eval] # filter based on what was queried
        if reg_strength>0:
            loss = loss_mean_kl(pp_approx,pp_target) + reg_strength*w_entropy(sppa)
        else:
            loss = loss_mean_kl(pp_approx,pp_target)
        loss.backward()
        return loss

    assert(x_eval.tolist()==list(range(1,101)))
    sppa = SparsePPApprox_NumGame()
    sppa = sppa.to(device=DEVICE)
    sppa.reg_strength = reg_strength
    params1 = [sppa.h_weight]
    params2 = [sppa.lapse_rate_weight]
    # params2 = [sppa.lapse_rate_weight, sppa.b_rate_weight]
    optimizer1 = optim.LBFGS(params1, lr=1., max_iter=200, line_search_fn='strong_wolfe')
    optimizer2 = optim.LBFGS(params2, lr=.05, max_iter=200, line_search_fn='strong_wolfe')
    sppa.train()
    prev_loss = float('inf')
    for step in range(nsteps):        
        optimizer1.step(closure)
        optimizer2.step(closure)
        myloss = closure().cpu().item()
        if verbose: print('after step',step,'loss',myloss)
        assert not torch.isnan(torch.tensor(myloss)), ' search stopped due to NaN loss'
        if not (myloss + step_tol < prev_loss):
            if verbose:
                print('search stopped due to no improvement')
                print(' lapse rate',sppa.lapse_rate.item())
                print(' base rate', sppa.b_rate.item())
            break
        prev_loss = myloss

    w = sppa.w.cpu().detach()
    pp_approx = sppa() # [101 x 2], log-probs
    pp_approx = pp_approx[x_eval] # filter based on what we used
    pp_approx = pp_approx.cpu().detach()
    sorted_idx = torch.argsort(w,descending=True)
    sorted_w = w[sorted_idx]
    sorted_names = [sppa.H_names[idx] for idx in sorted_idx]
    if verbose:
        ptitle = 'fit:'+sorted_names[0]+' w/ w='+str(round(sorted_w[0].item(),2))+'; '+sorted_names[1]+' w/ w='+str(round(sorted_w[1].item(),2))
        print('base rate', sppa.b_rate.item(), 'lapse rate', round(sppa.lapse_rate.item(),3))
        plot_pp_comparison(pp_approx, pp_target, title=ptitle)
        plt.show()
    return {'w' : w, 'pp_approx' : pp_approx, 'sorted_w' : sorted_w, 'sorted_names' : sorted_names}

def fit_sparse_pp_approx_set_concepts(pp_target, xq, list_H, nsteps=20, step_tol=1e-6, reg_strength =  5*1e-3, verbose=True):
    # Fit the sparse posterior predictive approximation for learning set of FOL concepts, minimizing KL and entropy
    # Input
    #   pp_target : [nobj x 2], target probabilities for posterior predictive (col 0 is yes, col 1 is no)
    #   xq : [nq list of sets], the queries for which we want to compute hypotheses
    #   reg_strength : strength of regularization
    # Output
    #   w : [tensor nhyp], the weight of each hypothesis
    #   pp_approx : [tensor nobj x 2], the posterior predictive probabilities of sparse approx
    #   sorted_w : [tensor nhyp], sorted weight of each hypothesis (descending)
    #   sorted_names : [list of str], sorted names of each hypothesis
    def closure():
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        pp_approx = sppa() # [nobj x 2], log-probs
        if reg_strength>0:
            loss = loss_mean_kl(pp_approx,pp_target) + reg_strength*w_entropy(sppa)
        else:
            loss = loss_mean_kl(pp_approx,pp_target)
        loss.backward()
        return loss

    sppa = SparsePPApprox_SetConcepts(xq, list_H)
    sppa = sppa.to(device=DEVICE)
    sppa.reg_strength = reg_strength
    # plot_pp_comparison(sppa(), pp_target, title='before')
    params1 = [sppa.h_weight]
    # params2 = [sppa.lapse_rate_weight, sppa.b_rate_weight]
    params2 = [sppa.lapse_rate_weight]
    optimizer1 = optim.LBFGS(params1, lr=1., max_iter=200, line_search_fn='strong_wolfe')
    optimizer2 = optim.LBFGS(params2, lr=.05, max_iter=200, line_search_fn='strong_wolfe')
    sppa.train()
    prev_loss = float('inf')
    for step in range(nsteps):        
        optimizer1.step(closure)
        optimizer2.step(closure)
        myloss = closure().cpu().item()
        if verbose: print('after step',step,'loss',myloss)
        assert not torch.isnan(torch.tensor(myloss)), ' search stopped due to NaN loss'
        if not (myloss + step_tol < prev_loss):
            if verbose:
                print('search stopped due to no improvement')
                print(' lapse rate',sppa.lapse_rate.item())
                print(' base rate', sppa.b_rate.item())
            break
        prev_loss = myloss

    w = sppa.w.cpu().detach()
    pp_approx = sppa() # [nobj x 2], log-probs
    pp_approx = pp_approx.cpu().detach()
    sorted_idx = torch.argsort(w,descending=True)
    sorted_w = w[sorted_idx]
    sorted_names = [sppa.H_names[idx] for idx in sorted_idx]
    if verbose:
        ptitle = 'fit:'+sorted_names[0]+' w/ w='+str(round(sorted_w[0].item(),2))+'; '+sorted_names[1]+' w/ w='+str(round(sorted_w[1].item(),2))
        print('base rate', sppa.b_rate.item(), 'lapse rate', round(sppa.lapse_rate.item(),3))
        plot_pp_comparison(pp_approx, pp_target, x_ticks_str=sppa.objects_flat, title=ptitle)
        plt.show()
    return {'w' : w, 'pp_approx' : pp_approx, 'sorted_w' : sorted_w, 'sorted_names' : sorted_names}

    
def plot_pp_comparison(pp_approx, pp_target, x_eval=None, x_ticks_str=[], title="Posterior Predictive Comparison"):
    """
    Plot a bar graph comparing pp_approx and pp_target.
    Args:
        pp_approx: tensor [N x 2], model probabilities (col 0: 'yes', col 1: 'no')
        pp_target: tensor [N x 2], target probabilities
        x_eval: list or array of indices/numbers (optional, for x-axis labels)
        x_ticks_str: list of strings for x-axis ticks (optional)
        title: str, plot title
    """
    pp_approx = pp_approx.detach().cpu().numpy()
    pp_target = pp_target.detach().cpu().numpy()
    N = pp_approx.shape[0]
    x = np.arange(N) if x_eval is None else np.array(x_eval)
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, pp_approx[:, 0], width, label='Approx Yes', alpha=0.7)
    plt.bar(x + width/2, pp_target[:, 0], width, label='Target Yes', alpha=0.7)
    if x_ticks_str:
        plt.xticks(x, x_ticks_str, rotation=90)
    plt.xlabel('Object for judgement')
    plt.ylabel('Probability (Yes)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def debug_prune_identical_hypotheses():
    # Build a SparsePPApprox_SetConcepts-like object manually,
    # bypassing __init__ to avoid needing real data.
    import types

    def make_obj(H_all, H_names, list_H):
        obj = object.__new__(SparsePPApprox_SetConcepts)
        nn.Module.__init__(obj)  # initialize Module so nn.Parameter works
        obj.H_all = H_all
        obj.H_names = H_names
        obj.list_H = list_H
        obj.h_weight = nn.Parameter(torch.ones(H_all.shape[0]))
        return obj

    # --- Main test: two pairs of duplicates + one unique ---
    # complexity_num_func counts '_(', so more function calls = higher complexity
    obj = make_obj(
        H_all=torch.tensor([
            [1., 0., 1.],  # row 0 — duplicate of row 2 (2 funcs, should be pruned)
            [0., 1., 0.],  # row 1 — duplicate of row 3 (1 func, should be kept)
            [1., 0., 1.],  # row 2 — duplicate of row 0 (1 func, should be kept)
            [0., 1., 0.],  # row 3 — duplicate of row 1 (3 funcs, should be pruned)
            [1., 1., 0.],  # row 4 — unique
        ], dtype=torch.float32),
        H_names=[
            'blue_(x) and green_(x)',       # 0 — 2 funcs, pruned (dup with row 2)
            'size1_(x)',                     # 1 — 1 func, kept (dup with row 3)
            'blue_(x)',                      # 2 — 1 func, kept
            'size1_(x) or size2_(x) or size3_(x)',  # 3 — 3 funcs, pruned
            'circle_(x)',                    # 4 — 1 func, unique, kept
        ],
        list_H=['h0', 'h1', 'h2', 'h3', 'h4'],
    )
    obj._prune_identical_hypotheses()

    # Verify counts
    assert obj.H_all.shape[0] == 3, f"Expected 3 hypotheses, got {obj.H_all.shape[0]}"
    assert len(obj.H_names) == 3
    assert len(obj.list_H) == 3
    assert obj.h_weight.shape[0] == 3

    # Verify we kept the simpler (fewer function calls) names
    assert 'blue_(x)' in obj.H_names, "Should keep 'blue_(x)' over 'blue_(x) and green_(x)'"
    assert 'size1_(x)' in obj.H_names, "Should keep 'size1_(x)' over 'size1_(x) or size2_(x) or size3_(x)'"
    assert 'circle_(x)' in obj.H_names, "Should keep unique hypothesis 'circle_(x)'"
    assert 'blue_(x) and green_(x)' not in obj.H_names
    assert 'size1_(x) or size2_(x) or size3_(x)' not in obj.H_names

    # Verify list_H tracks the same indices
    assert obj.list_H == ['h1', 'h2', 'h4'], f"Expected ['h1', 'h2', 'h4'], got {obj.list_H}"

    # Verify H_all rows match the kept hypotheses
    expected = torch.tensor([
        [0., 1., 0.],  # size1_(x)
        [1., 0., 1.],  # blue_(x)
        [1., 1., 0.],  # circle_(x)
    ])
    assert torch.equal(obj.H_all.cpu(), expected), f"H_all mismatch:\n{obj.H_all}"
    print("  Test 1 passed (two pairs of duplicates + one unique)")

    # --- Test 2: no duplicates, nothing should change ---
    obj2 = make_obj(
        H_all=torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float32),
        H_names=['blue_(x)', 'green_(x)', 'circle_(x)'],
        list_H=['h0', 'h1', 'h2'],
    )
    obj2._prune_identical_hypotheses()
    assert obj2.H_all.shape[0] == 3
    assert obj2.H_names == ['blue_(x)', 'green_(x)', 'circle_(x)']
    assert obj2.list_H == ['h0', 'h1', 'h2']
    print("  Test 2 passed (no duplicates)")

    # --- Test 3: all rows identical, keep the one with fewest function calls ---
    obj3 = make_obj(
        H_all=torch.tensor([[1., 1.], [1., 1.], [1., 1.]], dtype=torch.float32),
        H_names=['blue_(x) and green_(x)', 'blue_(x)', 'blue_(x) and green_(x) and circle_(x)'],
            # func counts:  2, 1, 3  → keep index 1
        list_H=['h0', 'h1', 'h2'],
    )
    obj3._prune_identical_hypotheses()
    assert obj3.H_all.shape[0] == 1
    assert obj3.H_names == ['blue_(x)'], f"Should keep simplest 'blue_(x)', got {obj3.H_names}"
    assert obj3.list_H == ['h1']
    print("  Test 3 passed (all identical, keep fewest function calls)")

    # --- Test 4: single hypothesis, nothing to prune ---
    obj4 = make_obj(
        H_all=torch.tensor([[0., 1., 0., 1.]], dtype=torch.float32),
        H_names=['circle_(x)'],
        list_H=['h0'],
    )
    obj4._prune_identical_hypotheses()
    assert obj4.H_all.shape[0] == 1
    assert obj4.H_names == ['circle_(x)']
    print("  Test 4 passed (single hypothesis)")

    print("All tests passed for _prune_identical_hypotheses")

def debug_flat_vs_numpy_mask():
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    mask = np.array([[True, False, True],
                     [False, True, False],
                     [True, True, False]])
    
    # flat approach: apply mask per row, then flatten
    masked_lists = [list(arr[i][mask[i]]) for i in range(arr.shape[0])]
    result_flat = flat(masked_lists)
    
    # numpy approach: boolean index directly (C-order)
    result_numpy = arr[mask].tolist()
    # print(masked_lists)
    # print(result_flat)
    assert result_flat == result_numpy, f"{result_flat} != {result_numpy}"
    print("test_flat_vs_numpy_mask passed")

if __name__ == '__main__':
    debug_flat_vs_numpy_mask()
    # debug_prune_identical_hypotheses()