from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from modules import MetaModuleMonteCarlo
from util import logit
import math

import scipy.special as sc


class VariationalApprox_v2(object):

    def __init__(self, device, num_mc_sample, model=None, mean_init=None, covar_init=None, 
                init_optim_lrsch=True, optim_outer_name=None, optim_outer_kwargs=None, 
                lr_sch_outer_name=None, lr_sch_outer_kwargs=None):

        assert (model is not None) or (mean_init is not None and covar_init is not None)

        self.num_mc_sample = num_mc_sample
        self.device = device
        
        # initialize the mean and variance of parameter
        self.mean = self.init_mean(model, mean_init)
        self.covar = self.init_covariance(model, covar_init)
        
        # detach the parameter depending on model or parameter
        if model is not None:
            self.detach_model_params(model)
        else:
            self.detach_mean_covar_params(mean_init, covar_init)
        
        # define optimiser and lr scheduler
        if init_optim_lrsch:
            self.optimizer = getattr(optim, optim_outer_name)\
                (list(self.mean.values()) + list(self.covar.values()), **optim_outer_kwargs)

            if lr_sch_outer_name is None:
                self.lr_scheduler = None
            else:
                self.lr_scheduler = getattr(lr_scheduler, lr_sch_outer_name)\
                    (self.optimizer, **lr_sch_outer_kwargs)
        else:
            self.optimizer = None
            self.lr_scheduler = None


    def init_mean(self, model, mean_init):

        if model is not None:
            return OrderedDict([
                (name, param.clone().detach().to(device=self.device).requires_grad_(True)) \
                    for (name, param) in model.meta_named_parameters()
            ])
        else:
            return OrderedDict([
                (name, param.clone().detach().to(device=self.device).requires_grad_(True)) \
                    for (name, param) in mean_init.items()
            ])


    def init_covariance(self, model, covar_init):

        if model is not None:
            return OrderedDict([
                (name, param.new_full(param.size(), fill_value=-10.).to(device=self.device).requires_grad_(True)) \
                    for (name, param) in model.meta_named_parameters()
            ])
            # create additional parameter for covariance
        else:
            return OrderedDict([
                (name, param.clone().detach().to(device=self.device).requires_grad_(True)) \
                    for (name, param) in covar_init.items()
            ])

    def sample_params(self, n_sample=None, detach_mean_cov=False):
        params = OrderedDict()
        for (name, mean), cov in zip(self.mean.items(), self.exp_covar(self.covar).values()):
            if n_sample == 1:
                params_sample_size = [*mean.size()]
            elif n_sample is None:
                params_sample_size = [self.num_mc_sample, *mean.size()]
            else:
                params_sample_size = [n_sample, *mean.size()]

            params[name] = \
                (mean.detach() + cov.detach().sqrt()
                 * torch.randn(*params_sample_size, dtype=mean.dtype, device=mean.device)).requires_grad_(True) \
                    if detach_mean_cov \
                else mean + cov.sqrt() * torch.randn(*params_sample_size, dtype=mean.dtype, device=mean.device)

        return params
    
    def detach_model_params(self, model):
        for param in model.meta_parameters():
            param.requires_grad = False
    
    def detach_mean_covar_params(self, mean, covar):
        for mean_tensor in mean.values():
            mean_tensor.requires_grad = False
        for covar_tensor in covar.values():
            covar_tensor.requires_grad = False

    def update_mean_cov(self):
        self.mean_old = OrderedDict([(name, mu.clone().detach()) for (name, mu) in self.mean.items()])
        self.covar_old = OrderedDict([(name, cov.clone().detach()) for (name, cov) in self.covar.items()])


class var_approx_beta(MetaModuleMonteCarlo):

    def __init__(self, alpha, k, num_mc_sample, device, 
                init_optim_lrsch=True, optim_outer_name=None, optim_outer_kwargs=None, 
                lr_sch_outer_name=None, lr_sch_outer_kwargs=None, implicit=True):
        super(var_approx_beta,self).__init__()
        """
        alpha: initial prior
        k: K-truncated version of stick-breaking
        """
        self.init_alpha = alpha
        self.device = device
        if isinstance(self.init_alpha, int):
            self.prior_alpha1 = torch.tensor(np.ones([k]), device=device).float()
            self.prior_alpha2 = torch.tensor(np.tile([self.init_alpha], [k]), device=device).float()
        else:
            # assign the previous value
            assert alpha.shape[1] == k
            self.prior_alpha1 = torch.tensor(self.init_alpha[0], device=device).float()
            self.prior_alpha2 = torch.tensor(self.init_alpha[1], device=device).float()
        
        if isinstance(self.init_alpha, int):
            self.var_gamma1 = nn.Parameter(torch.tensor(np.ones([k]), device=self.device).float())
            self.var_gamma2 = nn.Parameter(torch.tensor(np.tile([self.init_alpha], [k]), device=self.device).float())
        else:
            # assign the previous value
            assert alpha.shape[1] == k
            self.var_gamma1 = nn.Parameter(torch.tensor(self.init_alpha[0], device=self.device).float())
            self.var_gamma2 = nn.Parameter(torch.tensor(self.init_alpha[1], device=self.device).float())

        # init the bern
        # self.bern_pro = nn.Parameter(logit(torch.ones([k], device=self.device)*0.5))
        if init_optim_lrsch:
            self.optimizer = getattr(optim, optim_outer_name)\
                (self.parameters(), **optim_outer_kwargs)

            if lr_sch_outer_name is None:
                self.lr_scheduler = None
            else:
                self.lr_scheduler = getattr(lr_scheduler, lr_sch_outer_name)\
                    (self.optimizer, **lr_sch_outer_kwargs)
        else:
            self.optimizer = None
            self.lr_scheduler = None

        self.k = k
        self.num_mc_sample = num_mc_sample
        self.implicit = implicit
        self.device = device
    
    def update_posterior(self, k):
        
        if self.var_gamma1 is not None:
            # self.prior_alpha1 += self.var_gamma1.data[:self.k].detach()
            # self.prior_alpha2 += self.var_gamma2.data[:self.k].detach()
            # self.prior_alpha1 -= torch.ones_like(self.prior_alpha1).to(self.device)
            # self.prior_alpha2 -= torch.ones_like(self.prior_alpha2).to(self.device)
            self.reset_prior()

        # initialize the posterior for the next time
        if isinstance(self.init_alpha, int):
            self.var_gamma1 = nn.Parameter(torch.tensor(np.ones([k]), device=self.device).float())
            self.var_gamma2 = nn.Parameter(torch.tensor(np.tile([self.init_alpha], [k]), device=self.device).float())
        else:
            # assign the previous value
            assert self.init_alpha.shape[1] == k
            self.var_gamma1 = nn.Parameter(torch.tensor(self.init_alpha[0], device=self.device).float())
            self.var_gamma2 = nn.Parameter(torch.tensor(self.init_alpha[1], device=self.device).float())
    
    def reset_prior(self, top_k=3):
        '''
        throw the low one
        '''
        top_k = math.ceil(0.6*self.k)

        index = np.argsort(self.var_gamma1.data.cpu().numpy())[-top_k:]

        for index_ in index:
            self.prior_alpha2[index_] += 1
        
        index = np.argsort(self.var_gamma1.data.cpu().numpy())[:top_k]
        for index_ in index:
            self.prior_alpha1[index_] += 1


    # def update_bern(self):
    #     self.bern_pro = nn.Parameter(logit(torch.ones([self.k], device=self.device)*0.5))
    
    # def update_prior(self):

    #     self.prior_alpha1 = self.var_gamma1
    #     self.prior_alpha2 = self.var_gamma2
    
    def check(self):
        with torch.no_grad():
            self.var_gamma2 = nn.Parameter(torch.where(self.var_gamma2>1.0, self.var_gamma2, torch.tensor(1.0).to(self.device)))

    def set_gamma(self, gamma1, gamma2):
        
        self.var_gamma1 = gamma1
        self.var_gamma2 = gamma2
    
    def sample(self, size=None):
        if self.implicit:
            
            beta_distribution = torch.distributions.Beta(
                concentration1=self.var_gamma1, 
                concentration0=self.var_gamma2)
            
            if size is None:
                return beta_distribution.rsample()
            else:
                return beta_distribution.rsample(size)
        else:
            eps = 10e-8
            rand_u = torch.rand_like(self.var_gamma1)

            return torch.exp((1. / (self.var_gamma1 + eps)) \
                * torch.log(1. - torch.exp((1. / (self.var_gamma2 + eps)) \
                * torch.log(rand_u)) + eps))
    
    def sample_mean(self, size=None):
        
        return self.var_gamma1 / (self.var_gamma1+self.var_gamma2)
    
    def sample_prior_pros(self, size=None):

        if self.implicit:
            beta_distribution = torch.distributions.Beta(
                concentration1=self.prior_alpha1, 
                concentration0=self.prior_alpha2)
            
            if size is None:
                return beta_distribution.sample()
            else:
                return beta_distribution.sample(size)
        else:
            eps = 10e-8
            rand_u = torch.rand_like(self.prior_alpha1)

            return torch.exp((1. / (self.prior_alpha1 + eps)) \
                * torch.log(1. - torch.exp((1. / (self.prior_alpha2 + eps)) \
                * torch.log(rand_u)) + eps))

    def KL_terms(self):
        if self.implicit:

            euler_const = -torch.digamma(torch.tensor(1.0))

            conc1, conc2 = self.softplus(self.var_gamma1), self.softplus(self.var_gamma2)

            eps = 10e-8
            a_numpy = self.prior_alpha1.cpu().detach().numpy()
            b_numpy = np.ones_like(a_numpy)
            v_kl1 = ((conc1 - self.prior_alpha1)/(conc1+eps))*(-euler_const -torch.digamma(conc2) - 1.0/(conc2+eps))
            v_kl2 = ((conc1+eps).log() + (conc2+eps).log()) + torch.log(eps + torch.tensor(sc.beta(a_numpy,b_numpy))).to(self.device)
            v_kl3 = -(conc2 - 1)/(conc2+eps) 
            v_kl4 = torch.tensor(0.0).to(self.device)
            kl_beta = sum(v_kl1+v_kl2+v_kl3+v_kl4)

            return kl_beta
        else:
            kl = 1. / (1 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(1. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (2 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(2. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (3 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(3. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (4 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(4. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (5 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(5. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (6 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(6. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (7 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(7. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (8 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(8. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (9 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(9. / self.var_gamma1, self.var_gamma2)
            kl += 1. / (10 + self.var_gamma1 * self.var_gamma2) * self.beta_fn(10. / self.var_gamma1, self.var_gamma2)
            kl *= (self.prior_alpha2 - 1) * self.var_gamma2

            # use another taylor approx for Digamma function
            psi_b_taylor_approx = torch.log(self.var_gamma2) - 1. / (2 * self.var_gamma2) - 1. / (12 * self.var_gamma2 ** 2)
            kl += (self.var_gamma1 - self.prior_alpha1) / self.var_gamma1 * (-0.57721 - psi_b_taylor_approx - 1 / self.var_gamma2)  # T.psi(self.posterior_b)

            # add normalization constants
            kl = kl + torch.log(self.var_gamma1 * self.var_gamma2) + torch.log(self.beta_fn(self.prior_alpha1, self.prior_alpha2))

            # final term
            kl = kl - (self.var_gamma2 - 1) / self.var_gamma2

            return torch.sum(kl)

    def softplus(self, x, beta = 1.0, threshold = 20.0):
        return F.softplus(x, beta=beta, threshold=threshold)

    def beta_fn(self, a,b):
        return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b))

    def add_cluster(self, k):
        """
        add the cluster according to the required number
        """
        
        # if isinstance(self.init_alpha, int):
        #     extend_gamma1 = torch.tensor(np.ones([k]), device=self.device).float()
        #     extend_gamma2 = torch.tensor(np.tile([self.init_alpha], [k]), device=self.device).float()
        # else:
        #     # assign the previous value
        #     assert self.alpha.shape[1] == k
        #     extend_gamma1 = torch.tensor(self.init_alpha[0], device=self.device).float()
        #     extend_gamma2 = torch.tensor(self.init_alpha[1], device=self.device).float()
        extend_gamma1 = torch.tensor(np.ones([k]), device=self.device).float()
        extend_gamma2 = torch.tensor(np.tile([self.init_alpha], [k]), device=self.device).float()
        
        self.prior_alpha1 = torch.cat([self.prior_alpha1, extend_gamma1], 0)
        self.prior_alpha2 = torch.cat([self.prior_alpha2, extend_gamma2], 0)
        self.var_gamma1 = torch.nn.Parameter(torch.cat([self.var_gamma1.data, extend_gamma1],0))
        self.var_gamma2 = torch.nn.Parameter(torch.cat([self.var_gamma2.data, extend_gamma2],0))

        self.k += k

    def delete_cluster(self, indexs):
        """
        delete the cluster according to the indexes
        input: indexs - an input np.array, consisting of one and zero
        """
        delete_index = np.where(indexs==0)[0]

        # for posterior
        np_var_gamma1 = np.delete(self.var_gamma1.data.cpu().numpy(), delete_index)
        self.var_gamma1 = torch.nn.Parameter(torch.tensor(np_var_gamma1).to(self.device))

        np_var_gamma2 = np.delete(self.var_gamma2.data.cpu().numpy(), delete_index)
        self.var_gamma2 = torch.nn.Parameter(torch.tensor(np_var_gamma2).to(self.device))

        # for prior
        np_alpha1 = np.delete(self.prior_alpha1.cpu().numpy(), delete_index)
        self.prior_alpha1 = torch.tensor(np_alpha1).to(self.device)

        np_alpha2 = np.delete(self.prior_alpha2.cpu().numpy(), delete_index)
        self.prior_alpha2 = torch.tensor(np_alpha2).to(self.device)

        self.k -= len(delete_index)
        

if __name__ == '__main__':

    aaa = var_approx_beta(1,3,1,device='cpu', optim_outer_name='Adam')
    aaa.reset_prior()