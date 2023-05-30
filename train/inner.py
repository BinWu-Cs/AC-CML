import torch
import torch.nn.functional as F
from collections import OrderedDict

from functional.cross_entropy import cross_entropy
from train.util import get_accuracy, kldiv_mvn_diagcov, exp_covar


def inner_maml_amvi(model, var_obj_list, z_t, inputs, labels, nstep_inner=5, lr_inner=0.4, \
                    first_order=False, device=None, kl_scale=0.01, sample_batch=False):
    '''
    inner update to get the specific parameters
    '''
    # zero grad
    for var_obj in var_obj_list:
        for mu, cov in zip(var_obj.mean.values(), var_obj.covar.values()):
            if mu.grad is not None and cov.grad is not None:
                mu.grad.zero_()
                cov.grad.zero_()

    batch_size = inputs.size()[1]
    num_batch_per_inner_step = 1 if sample_batch else var_obj_list[0].num_mc_sample

    # initialize model parameters
    mean_merge = OrderedDict([(name, 0.0) for (name, param) in var_obj.mean.items()])
    covar_merge = OrderedDict([(name, 0.0) for (name, param) in var_obj.covar.items()])

    n = len(mean_merge)

    for name, _ in var_obj_list[0].mean.items():

        for i in range(len(var_obj_list)):
            var_obj = var_obj_list[i]
            mean_merge[name] += var_obj.mean[name] * z_t[i]
            covar_merge[name] += var_obj.covar[name] * z_t[i]

    mean_inner = OrderedDict([(name, param.clone()) for (name, param) in mean_merge.items()])
    covar_inner = OrderedDict([(name, param.clone()) for (name, param) in covar_merge.items()])

    for _ in range(nstep_inner):
        
        # compute the loss of each inner step
        outputs = model(inputs, mean=mean_inner, cov=exp_covar(covar_inner))
        nll = cross_entropy(outputs, labels, reduction='mean') 

        # for _ in range(var_obj.num_mc_sample):
        for __ in range(num_batch_per_inner_step - 1):
            outputs = model(inputs, mean=mean_inner, cov=exp_covar(covar_inner))
            nll += cross_entropy(outputs, labels, reduction='mean')

        # the kl divergence between the phi and theta
        kl = kldiv_mvn_diagcov(
                mean_p=mean_inner, cov_p=exp_covar(covar_inner),
                mean_q=mean_merge, cov_q=exp_covar(mean_merge)
            ) / (batch_size * var_obj.num_mc_sample)

        # loss = nll/var_obj.num_mc_sample + kl_scale * kl 
        loss = nll / num_batch_per_inner_step + kl_scale * kl

        # torch.autograd.grad do not accumulate gradients to tensor.grad
        grads = torch.autograd.grad(loss, list(mean_inner.values())+list(covar_inner.values()), create_graph=not first_order)

        mean_inner = OrderedDict([(name, param - lr_inner * grad) for (name, param), grad in zip(mean_inner.items(), grads[:n])])
        covar_inner = OrderedDict([(name, param - lr_inner * grad) for (name, param), grad in zip(covar_inner.items(), grads[n:])])
    
    return mean_inner, covar_inner

def inner_maml_amvi_v2(model, var_obj, inputs, labels, nstep_inner=5, lr_inner=0.4, \
                    first_order=False, device=None, kl_scale=0.01, sample_batch=False):
    '''
    inner update to get the specific parameters
    '''
    for mu, cov in zip(var_obj.mean.values(), var_obj.covar.values()):
        if mu.grad is not None and cov.grad is not None:
            mu.grad.zero_()
            cov.grad.zero_()

    batch_size = inputs.size()[1]
    n = len(var_obj.mean)
    num_batch_per_inner_step = 1 if sample_batch else var_obj.num_mc_sample

    # initialize model parameters
    mean_inner = OrderedDict([(name, param.clone()) for (name, param) in var_obj.mean.items()])
    covar_inner = OrderedDict([(name, param.clone()) for (name, param) in var_obj.covar.items()])

    for _ in range(nstep_inner):
        
        # compute the loss of each inner step
        outputs = model(inputs, mean=mean_inner, cov=exp_covar(covar_inner))
        nll = cross_entropy(outputs, labels, reduction='mean') 

        # for _ in range(var_obj.num_mc_sample):
        for __ in range(num_batch_per_inner_step - 1):
            outputs = model(inputs, mean=mean_inner, cov=exp_covar(covar_inner))
            nll += cross_entropy(outputs, labels, reduction='mean')

        # the kl divergence between the phi and theta
        kl = kldiv_mvn_diagcov(
                mean_p=mean_inner, cov_p=exp_covar(covar_inner),
                mean_q=var_obj.mean, cov_q=exp_covar(var_obj.covar)
            ) / (batch_size * var_obj.num_mc_sample)

        # loss = nll/var_obj.num_mc_sample + kl_scale * kl 
        loss = nll / num_batch_per_inner_step + kl_scale * kl

        # torch.autograd.grad do not accumulate gradients to tensor.grad
        grads = torch.autograd.grad(loss, list(mean_inner.values())+list(covar_inner.values()), create_graph=not first_order)

        mean_inner = OrderedDict([(name, param - lr_inner * grad) for (name, param), grad in zip(mean_inner.items(), grads[:n])])
        covar_inner = OrderedDict([(name, param - lr_inner * grad) for (name, param), grad in zip(covar_inner.items(), grads[n:])])
    
    return mean_inner, covar_inner