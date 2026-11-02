import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.special import logsumexp

mylambda_default = 2./3 # weight on mathematical hypotheses (vs. interval)
    # value of lambda and signa set by eye, from Tenenbaum's thesis
x_max = 100 # (numbers between 1 and 100 are allowed)	
x_all = np.arange(0,x_max+1) # include 0 for convenience, although it's not included in any hypothesis
x_eval = [2,4,6,8,9,10]+list(range(12,23))+[24,25,26,28,32,36,41,56,62,64,87,95,96] # numbers evaluated in human experiments

# Generate a list of all mathematical hypotheses
def make_h_odd():
    return list(range(1,x_max+1,2)), 'odds'

def make_h_even():
    return list(range(2,x_max+1,2)), 'evens'

def make_h_square():
    h = []
    for x in range(1,x_max+1):
        if x**2 <= x_max:
            h.append(x**2)
    return h, 'squares'

def make_h_cube():
    h = []
    for x in range(1,x_max+1):
        if x**3 <= x_max:
            h.append(x**3)
    return h, 'cubes'

def make_h_primes():
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97], 'primes'

def make_h_mult_of_y(y):
    h = []
    for x in range(1,x_max+1):
        if x*y <= x_max:
            h.append(x*y)
    return h, 'mults_of_'+str(y)

def make_h_powers_of_y(y):
    h = []
    for x in range(1,x_max+1):
        if y**x <= x_max:
            h.append(y**x)
    return h, 'powers_of_'+str(y)

def make_h_numbers_ending_in_y(y):
    h = []
    for x in range(1,x_max+1):
        if str(x)[-1] == str(y):
            h.append(x)
    return h, 'ending_in_'+str(y)

def generate_math_hypotheses(mylambda):
    # mylambda is the weight of interval hypotheses
    h_pairs = [make_h_odd(), make_h_even(), make_h_square(), make_h_cube(), make_h_primes()]
    h_pairs += [make_h_mult_of_y(y) for y in range(3,13)]
    h_pairs += [make_h_powers_of_y(y) for y in range(2,11)]
    h_pairs += [make_h_numbers_ending_in_y(y) for y in range(0,10)]
    h_set = [pair[0] for pair in h_pairs]
    h_names = [pair[1] for pair in h_pairs]
    n_hyp = len(h_set)
    log_prior = np.log(mylambda * np.ones(n_hyp) / float(n_hyp))
    return h_set, log_prior, h_names

## Generate a list of all interval hypotheses
def make_h_between_y_and_z(y,z):
    assert(y >= 1 and z <= x_max)
    return list(range(y,z+1)), '[' + str(y) + ',...,' + str(z) + ']'

def pdf_erlang(x,sigma=10.):
    return (x / sigma**2) * math.exp(-x/sigma)

def generate_interval_hypotheses(mylambda):
    # 1-mylambda is the weight of interval hypotheses
    h_pairs = []
    for y in range(1,x_max+1):
        for z in range(y,x_max+1):            
            h_pairs.append(make_h_between_y_and_z(y,z))
    h_set = [pair[0] for pair in h_pairs]
    h_names = [pair[1] for pair in h_pairs] 
    nh = len(h_set)
    pv = np.ones(nh)
    for idx,h in enumerate(h_set): # prior based on length
        pv[idx] = pdf_erlang(len(h))
    pv = pv / np.sum(pv)
    pv = (1-mylambda) * pv
    log_prior = np.log(pv)
    return h_set, log_prior, h_names

def convert_h_list_to_numpy(h_list):
    h_numpy = np.zeros(x_all.size)
    if len(h_list) > 0: h_numpy[np.array(h_list)] = 1
    return h_numpy

def convert_x_list_to_numpy(x_list):
    x_numpy = np.zeros(x_all.size)
    if len(x_list) > 0: 
        for x in x_list:
            x_numpy[x] += 1
    return x_numpy

def convert_numpy_to_h_list(h_numpy):
	return np.nonzero(h_numpy)[0].tolist()

def generate_hypotheses(mylambda):
    h_math,lp_math,names_math = generate_math_hypotheses(mylambda)
    h_interval,lp_interval,names_interval = generate_interval_hypotheses(mylambda)
    H = h_math + h_interval
    H_names = names_math + names_interval
    H_numpy = [convert_h_list_to_numpy(h) for h in H]
    log_prior = np.concatenate((lp_math,lp_interval))
    assert(np.isclose(np.sum(np.exp(log_prior)),1.0))
    return H_numpy,log_prior,H_names

def list_of_num_to_hname(x):
    h_math,lp_math,names_math = generate_math_hypotheses(mylambda_default)
    h_interval,lp_interval,names_interval = generate_interval_hypotheses(mylambda_default)
    if x in h_math:
        return names_math[h_math.index(x)]
    if x in h_interval:
        return names_interval[h_interval.index(x)]
    assert('hypothesis not found')

def log_posterior(data,list_hypothesis,log_prior):
    # INPUT
    #  data : python list of observed numbers (X) 
    #  list_hypothesis : [nh length list] each hypothesis is a binary numpy array 
    #  log_prior : numpy vector [length nh] (log prior value for each hypothesis)
    # RETURN
    #  lpost : log-posterior value [length nh]
    # data_numpy = convert_h_list_to_numpy(data) # length nh numpy vector
    data_numpy = convert_x_list_to_numpy(data) # length 101 numpy vector, with count of each number
    nh = len(list_hypothesis)
    ll = np.zeros(nh)
    for idx,h in enumerate(list_hypothesis):
        ll[idx] = log_likelihood(data_numpy,h)
    lpost = ll + log_prior
    lpost = lpost - logsumexp(lpost)
    return lpost

def log_likelihood(data_numpy, hypothesis):
    # INPUT
    #  data_numpy : size x_max (numpy array) (count for each observer number)
    #  hypothesis: size x_max binary numpy array (included numbers in single hypothesis)
    # RETURN
    #  ll : log-likelihood value
    assert(hypothesis.size == data_numpy.size)
    n_d = np.sum(data_numpy)
    n_h = np.sum(hypothesis)
    if not np.isclose(np.dot(hypothesis,data_numpy), n_d):
        return -np.inf
    ll = n_d * np.log(1./n_h)
    return ll

def bayesian_predictions(data_eval, data, list_hypothesis, log_prior):
    # INPUT
    #  data_eval : [length ne python list] of new numbers we want to check the probability of membership for
    #     each number in data_eval is to be evaluated independently in the posterior predictive
    #  data : [python list] observed numbers (X) 
    #  list_hypothesis : python list of hypotheses, each is a binary numpy array 
    #  log_prior : numpy vector [length nh] which is the log prior value for each hypothesis
    # 
    # RETURN
    #  pp : numpy vector [size ne] of predicted probabilities of new numbers in data_eval (NOTE: NOT IN LOG SPACE)
    lpost = log_posterior(data,list_hypothesis,log_prior)
    post = np.exp(lpost) # posterior probabilities
    h_mat = np.array(list_hypothesis) # create a [nh by x_max] numpy matrix, showing numbers in each hypothesis
    ne = len(data_eval) # how many numbers to evaluate
    pp = np.zeros(ne) # predicted probability of each number
    for idx,de in enumerate(data_eval):
        pp[idx] = np.sum(h_mat[:,de] * post)
    return pp

def plot_predictions(x_eval,mypred):
	# INPUT
	#  x_eval : x-values in the plot
	#  mypred : corresponding value for each x belonging in the concept, according to posterior predictive
    mybottom = -0.1
    plt.figure(figsize=(10, 2))
    plt.bar(x_eval,mypred-mybottom,bottom=mybottom)
    plt.ylim((mybottom,1.2))
    plt.xticks(np.arange(0, x_max+1, step=4))
    plt.yticks([0,0.5,1])

def subplot_predictions(ax,x_eval,mypred, highlight=[]):
    # INPUT
    #  ax : subplot axis
    #  x_eval : x-values in the plot
    #  mypred : corresponding value for each x belonging in the concept, according to posterior predictive
    #  highlight : list of x-values to highlight
    mybottom = -0.1
    ax.bar(x_eval,mypred-mybottom,bottom=mybottom,color=(14/255., 13/255., 153/255., 1.0))
    sub_x_eval = np.array([x for idx,x in enumerate(x_eval) if x in highlight])
    sub_mypred = np.array([mypred[idx] for idx,x in enumerate(x_eval) if x in highlight])
    ax.bar(sub_x_eval,sub_mypred-mybottom,bottom=mybottom,color='orange')
    ax.set_ylim((mybottom,1.2))
    ax.set_xticks(np.arange(0, x_max+1, step=4))
    ax.set_yticks([0,0.5,1])

def forward_predictive_model(nsamp_hyp, nsamp_support, nsamp_query, mylambda=mylambda_default):
    # Draw samples from the forward model
    #
    # INPUT
    #  nsamp_hyp : number of hypotheses to be sample from the prior
    #
    # RETURN
    #  list_H : [nsamp length python list] of sampled hypotheses (each a python list of numbers)

    # sample p(h)
    H_all,log_prior_all,H_names = generate_hypotheses(mylambda)
    H_all_list = [convert_numpy_to_h_list(h) for h in H_all]
    N_H = len(H_all_list)
    list_H = random.choices(H_all_list, k=nsamp_hyp, weights = np.exp(log_prior_all)) # with replacement
    	# [nsamp_hyp len python list] of sampled hypotheses (each a python list)

    samples = []
    for h in list_H: # for each of the sample hypothesis

        # sample p(x|h)
        xs = random.choices(h, k=nsamp_support) # with replacement
        ys = ['yes']*len(xs)

        # sample arbitrary x_q from x_all and check if x_q \in h
        xq = random.choices(x_all[1:], k=nsamp_query)  # with replacement, excluding x=0
        yq = ['yes' if x in h else 'no' for x in xq] # check if each item is in the hypothesis
        S = {'xs':xs, 'ys':ys, 'xq':xq, 'yq':yq, 'aux':{'h_str':str(h)}}
        samples.append(S)
    return samples

def forward_model_special(nsamp_support, hyp_name='', mylambda=mylambda_default):
    # Draw samples from the forward model for particular hypothesis 'hyp_name'
    # Also, include all possible queries
    #
    # INPUT
    # nsamp_support : number of support examples from the likelihood
    # hyp_name : name of hypothesis to sample from (default is random from prior)
    #
    # RETURN
    #  S : structure containing xs, ys, xq, yq, and aux
    H_all,log_prior_all,H_names = generate_hypotheses(mylambda)
    H_all_list = [convert_numpy_to_h_list(h) for h in H_all]
    if hyp_name:
        h = H_all_list[H_names.index(hyp_name)]
    else:
        hyp_name = random.choices(H_names, k=1, weights = np.exp(log_prior_all))[0] # with replacement
        h = H_all_list[H_names.index(hyp_name)]
    
    # sample p(x|h)
    xs = random.choices(h, k=nsamp_support) # with replacement
    ys = ['yes']*len(xs)

    # sample arbitrary x_q from x_all and check if x_q \in h
    xq = x_all[1:] # with replacement, excluding x=0
    yq = ['yes' if x in h else 'no' for x in xq] # check if each item is in the hypothesis
    S = {'xs':xs, 'ys':ys, 'xq':xq, 'yq':yq, 'aux':{'h_str':str(h) ,'h_name':hyp_name}}
    return S

def get_top_hypotheses(data, list_hypothesis, log_prior, H_names, n=5):
    # INPUT
    #  data : python list of observed numbers (X) 
    #  list_hypothesis : [nh length list] each hypothesis is a binary numpy array 
    #  log_prior : numpy vector [length nh] (log prior value for each hypothesis)
    #  n : number of top hypotheses to return
    # RETURN
    #  top_hypotheses : list of top n hypotheses
    lpost = log_posterior(data,list_hypothesis,log_prior)
    post = np.exp(lpost) # posterior probabilities
    top_idx = np.argsort(post)[::-1][:n]
    top_hypotheses = [(H_names[idx],round(post[idx],3)) for idx in top_idx]

    # the posterior entropy
    non_zero_post = post[post > 0]
    post_entropy = -np.sum(non_zero_post * np.log(non_zero_post))
    print('Posterior entropy:',round(post_entropy,3))
    return top_hypotheses

def compare_two_hypotheses_posterior(data, h1_name, h2_name, list_hypothesis, log_prior, H_names):
    # INPUT
    #  data : python list of observed numbers (X) 
    #  h1_name : name of hypothesis 1
    #  h2_name : name of hypothesis 2
    #  list_hypothesis : [nh length list] each hypothesis is a binary numpy array 
    #  log_prior : numpy vector [length nh] (log prior value for each hypothesis)
    lpost = log_posterior(data,list_hypothesis,log_prior)
    post = np.exp(lpost) # posterior probabilities
    idx1 = H_names.index(h1_name)
    idx2 = H_names.index(h2_name)
    post1 = post[idx1]
    post2 = post[idx2]
    print(h1_name,'posterior:', post1)
    print(h2_name,'posterior:', post2)
    print('ratio',post1 / post2)

if __name__ == '__main__':
    H_all,log_prior_all,H_names = generate_hypotheses(mylambda_default)
    
    print(get_top_hypotheses([66,78], H_all, log_prior_all, H_names, n=5))
    print(get_top_hypotheses([8,80,48], H_all, log_prior_all, H_names, n=5))

    # compare_two_hypotheses_posterior([31, 3, 1, 15], '[1,...,31]', '[1,...,39]', H_all, log_prior_all, H_names)


    # print('Number of hypotheses:', len(H_all))
    # mypred = bayesian_predictions(x_eval, [16], H_all, log_prior_all)
    # plot_predictions(x_eval,mypred)
    # plt.title('Bayesian inference, X=[16]')
    # plt.ylabel('prob. of membership')
    # mypred = bayesian_predictions(x_eval, [16, 8, 2, 64], H_all, log_prior_all)
    # plot_predictions(x_eval,mypred)
    # plt.title('Bayesian inference, X=[16, 8, 2, 64]')
    # mypred = bayesian_predictions(x_eval, [16, 23, 19, 20], H_all, log_prior_all)
    # plot_predictions(x_eval,mypred)
    # plt.title('Bayesian inference, X=[16, 23, 19, 20]')
    # plt.show()

    # H_all = np.array(H_all)
    # print('H_all shape:', H_all.shape)
    # hsize = H_all.sum(axis=1)
    # print('max hypothesis size:', hsize.max())
    # print('min hypothesis size:', hsize.min())