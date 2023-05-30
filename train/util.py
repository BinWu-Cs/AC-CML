import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math

import os
import random
from collections import OrderedDict
from PIL import Image

from data_generate import transformations as transfm

def split_path(path):
    # remove trailing '/' if any
    split_ls = os.path.normpath(path).split('/')
    if '' in split_ls:
        split_ls.remove('')
    if '.' in split_ls:
        split_ls.remove('.')
    return split_ls


def concat_param(weight, bias):
    if bias is not None:
        return torch.cat([weight, bias.unsqueeze(-1)], dim=-1)
    else:
        return weight


def enlist_transformation(img_resize=None, resize_interpolation='BILINEAR', is_grayscale=False, device=None, img_normalise=True):
    transform_ls = []
    if img_resize is not None:
        transform_ls.append(transforms.Resize(size=(img_resize, img_resize), interpolation=getattr(Image, resize_interpolation)))
    if is_grayscale:
        transform_ls.append(transforms.Grayscale())
    transform_ls.append(transforms.ToTensor())
    if device is not None:
        transform_ls.append(transfm.ToDevice(device=device))
    if img_normalise:
        transform_ls.append(transfm.NormaliseMinMax())
    return transform_ls


def enlist_montecarlo_param(param, num_mc_sample):
    return [OrderedDict([(name, params[i, ...]) for (name, params) in param.items()]) for i in range(num_mc_sample)]


def concat_montecarlo_param(list):
    return OrderedDict([(name, torch.stack([list[i][name] for i in range(len(list))], dim=0)) for name in list[0].keys()])


def kldiv_mvn_diagcov(mean_p, cov_p, mean_q, cov_q):
    kl_layer_ls = []
    for mu_p, sig_p, mu_q, sig_q in zip(mean_p.values(), cov_p.values(), mean_q.values(), cov_q.values()):
        mean_diff = mu_q - mu_p
        sig_q_inv = 1 / sig_q
        kl_layer = torch.log(sig_q).sum() - torch.log(sig_p).sum() - mu_p.numel() + (sig_q_inv * sig_p).sum() \
                   + ((mean_diff * sig_q_inv) * mean_diff).sum()
        kl_layer_ls.append(kl_layer)
    return sum(kl_layer_ls) / 2


def get_accuracy(labels, outputs=None, inputs=None, model=None, param=None):
    # outputs in (batch, numclass), softmax. or (n_sample, batch, numclass)
    # labels in (batch) or (n_sample, batch)
    # return the percentage of the accuracy
    if outputs is None:
        outputs = model(inputs, param=param)
    preds = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
    acc = torch.mean(preds == labels, dtype=torch.float32) * 100.
    return acc

def get_zk_likelihood(var_obj, prior_mean, prior_covar, pos_mean, pos_covar, var_beta, kth, device, batch_size, num_samples=1):

    """
    prior_mean_list = list(prior_mean.values())
    prior_covar_list = list(prior_covar.values())
    pos_mean_list = list(pos_mean.values())
    pos_covar_list = list(pos_covar.values())
    if num_samples > 1:
        sample_list = [mean + torch.exp(covar) * torch.randn_like(mean).expand(num_samples, *mean.size()) \
                    for mean, covar in zip(pos_mean_list, pos_covar_list)]
    else:
        sample_list = [mean + torch.exp(covar) * torch.randn_like(mean) for mean, covar in zip(pos_mean_list, pos_covar_list)]

    ll = torch.tensor(0.0, device=device)
    n = torch.tensor(0.0, device=device)

    for i in range(len(sample_list)):
        ll += 0.5 *1e-4 * torch.sum( 2*pos_covar_list[i] - 2*prior_covar_list[i] \
            + torch.square(sample_list[i]-pos_mean_list[i]) / torch.exp(2*pos_covar_list[i]) \
            - torch.square(sample_list[i]-prior_mean_list[i]) / torch.exp(2*prior_covar_list[i]) )
        n += torch.numel(sample_list[i])
    """
    kl = 0.01 * kldiv_mvn_diagcov(
        mean_p = pos_mean, cov_p=var_obj.exp_covar(pos_covar),
        mean_q=prior_mean, cov_q=var_obj.exp_covar(prior_covar)
    ) / (batch_size * num_samples)

    exp_pi = torch.tensor(1.0, device=device)
    for i in range(kth):
        # compute the pi according to the stick-breaking rules
        exp_pi = exp_pi * (1 - var_beta.var_gamma1[i].to(device=device) / \
                (var_beta.var_gamma1[i].to(device=device) + var_beta.var_gamma2[i].to(device=device)))
    if kth < var_beta.k - 1:
        exp_pi = exp_pi * var_beta.var_gamma1[kth].to(device=device) / \
                (var_beta.var_gamma1[kth].to(device=device) + var_beta.var_gamma2[kth].to(device=device))

    # exp_pi = 1e2 * exp_pi / (batch_size * num_samples)
    # exp_pi = exp_pi / (batch_size * num_samples)
    # print("Comparison between zk likelihood and expectation of pi:{}, {}".format(kl, exp_pi))

    return exp_pi - kl

def reparam_bernoulli(pros, temp=0.1):
    '''
    input: the pis
    output: the sampled z
    '''
    eps = 10e-8
    log_pro = torch.log(pros)
    rand_u = torch.rand_like(log_pro)
    l = torch.log(rand_u+eps) - torch.log(1-rand_u+eps)
    sample_bern = torch.sigmoid((l+log_pro)/temp)

    return sample_bern

def kl_bernoulli(pros, z_k, prior_pros, temp=0.1, prior_temp=10.0):
    '''
    compute the KL term of bernoulli distribution
    '''
    eps = 10e-8

    #log_pros = torch.log(pros)
    #log_prior_log = torch.log(prior_pros+eps)

    log_sample = torch.log(z_k+eps) - torch.log(1-z_k+eps)

    bern_kl1 = log_gumb(temp, pros, log_sample)
    bern_kl2 = log_gumb(prior_temp, prior_pros, log_sample)

    bern_kl = (bern_kl1 - bern_kl2).mean(0)
    
    return bern_kl

def log_gumb(temp, log_alpha, log_sample):
    ## Returns log probability of gumbel distribution
    eps = 10e-8
    exp_term = log_alpha + log_sample*(-temp)
    log_prob = exp_term + math.log(temp+eps) - 2*softplus(exp_term)
    return log_prob

def softplus(x, beta = 1.0, threshold = 20.0):
    return F.softplus(x, beta=beta, threshold=threshold)

def logit(x):
    eps = 10e-8
    return (x+eps).log() - (1-x+eps).log()

def exp_covar(covar):
    return OrderedDict([(name, torch.exp(cov)) for name, cov in covar.items()])
