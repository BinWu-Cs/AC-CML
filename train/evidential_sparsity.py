import math
import numpy as np

def evidential_sparsity(var_beta, previous_beta_alpha, previous_beta_beta, eta=0.5):
    """
    evidential sparsity for IBP
    """
    
    alphas = var_beta.var_gamma1.data.cpu().numpy()
    betas = var_beta.var_gamma2.data.cpu().numpy()

    # compute the positive and negtive weight
    weight = np.exp(alphas) - eta*np.exp(betas)
    pos_weight = np.maximum(weight,0)
    neg_weight = np.minimum(weight,0)

    for previous_alpha, previous_beta in zip(previous_beta_alpha, previous_beta_beta):

        previous_weight = np.exp(previous_alpha)-eta*np.exp(previous_beta)
        pre_pos_weight = np.maximum(previous_weight,0)
        pre_neg_weight = np.minimum(previous_weight,0)

        pos_weight += np.pad(pre_pos_weight, (0,len(pos_weight)-len(pre_pos_weight)), 'constant')
        neg_weight += np.pad(pre_neg_weight, (0,len(neg_weight)-len(pre_neg_weight)), 'constant')


    # compute the positive and negative term
    pos_term = np.exp(pos_weight) - 1
    neg_term = np.exp(neg_weight) - 1

    # compute the mass function
    results = []
    for i in range(len(pos_term)):
        neg_result = 1.
        for j in range(len(neg_term)):
            if i != j:
                neg_result *= neg_term[j]
        
        result = (neg_result + pos_term[i])*(np.exp(neg_weight)[i])
        results.append(result)
    
    # normalization
    np_results = np.array(results)
    np_results /= np_results.sum()

    return np_results