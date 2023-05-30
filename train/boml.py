import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.multinomial import Multinomial
from tqdm import trange
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import chain
from more_itertools import roundrobin

import numpy as np
import random
from functional.cross_entropy import cross_entropy
from data_generate.sampler import SuppQueryBatchSampler
from train.inner import inner_maml_amvi, inner_maml_amvi_v2
from train.util import get_accuracy, kldiv_mvn_diagcov, get_zk_likelihood, reparam_bernoulli, kl_bernoulli, exp_covar, logit
from evaluate.util_eval import meta_evaluation_amvi, meta_evaluation_amvi_v2, meta_evaluation_amvi_v3
from train.variational import VariationalApprox_v2
from train.evidential_sparsity import evidential_sparsity
from train import model as models



def metatrain_seqtask_multi_amvi(model, config, var_approx_list, var_beta, model_device, trainset, evalset, outer_kl_scale, 
        nstep_outer, nstep_inner, lr_inner, first_order, seqtask, num_task_per_itr, 
        task_by_supercls, num_way, num_shot, num_query_per_cls, eval_prev_task, eval_per_num_iter, 
        num_eval_task, eval_task_by_supercls, nstep_inner_eval, writer, prev_glob_step, verbose, sample_mean,sample_batch, max_eval, toe=False):

    '''
    train the model under the sequential task setting
    '''

    # set the sampler
    trainloader = []
    for one_trainset in trainset:
        trainsampler = SuppQueryBatchSampler(
            dataset=one_trainset, seqtask=False, num_task=num_task_per_itr, task_by_supercls=task_by_supercls,
            num_way=num_way, num_shot=num_shot, num_query_per_cls=num_query_per_cls
        )
        trainloader.append(DataLoader(one_trainset, batch_sampler=trainsampler))
    
    # define the list to save the previous veta
    var_previous_beta_alpha = []
    var_previous_beta_beta = []

    for time_ in trange(config['seq_task']):

        # determine the number of cluster
        if config['increase_num'] and time_ != 0:

            # Done: determine the number of cluster
            increase_num = 0
            if config['k'] + config['increase_k'] <= config['max_k']:
                increase_num = config['increase_k']
            elif config['k'] < config['max_k']:
                increase_num = config['max_k'] - config['k']
            
            if increase_num != 0:
                config['k'] += increase_num
                
                # add the cluster of beta
                var_beta.add_cluster(increase_num)

                # add the cluster of theta
                for _ in range(increase_num):
                    model_temp = getattr(models, config['net'])(**config['net_kwargs'])
                    approx = VariationalApprox_v2(
                        config['device'], config["num_mc_sample"], model=model_temp, init_optim_lrsch=False
                    )
                    approx.update_mean_cov()
                    var_approx_list.append(approx)
        # determine the dataset
        dataset_index = random.randint(0,config['num_dataset_to_run']-1)

        test_num = 0
        for batch_idx, (images, labels) in enumerate(trainloader[dataset_index], 0):
            
            # the optimization for each task
            for itr in range(config['one_task_itr']):
                negloglik_supp, negloglik_query, loss_outer, accuracy_query = \
                outer_step_seqtask(var_approx_list, model, var_beta, images, labels, num_way, num_shot, num_query_per_cls,\
                        outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, model_device, sample_mean, sample_batch, seq_task=True)
                # record the training information
                writer.add_scalar(
                        tag='loss_meta_train_task',
                        scalar_value=loss_outer, global_step=prev_glob_step + itr
                    )
            
                writer.add_scalar(
                        tag='accuracy_meta_train_task',
                        scalar_value=accuracy_query, global_step=prev_glob_step + itr
                    )
            
            test_num += 1
            print(test_num)

            
        # for evaluation
        for ldr_idx, evset in enumerate(evalset):
            if not max_eval:
                loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi_v2(
                evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_way=num_way,
                num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=model,
                variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval,
                lr_inner=lr_inner, model_device=model_device, var_beta=var_beta, sample_batch=sample_batch
                )
            else:
                loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi_v3(
                evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_way=num_way,
                num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=model,
                variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval,
                lr_inner=lr_inner, model_device=model_device, var_beta=var_beta, sample_batch=sample_batch
                )

            # Done: save the related losses
            writer.add_scalar(
                tag='loss_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss_meta_eval',
                scalar_value=loss_eval, global_step=time_
            )
            writer.add_scalar(
                tag='accuracy_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'accuracy_meta_eval',
                scalar_value=accuracy_eval, global_step=time_
            )
            writer.add_scalar(
                tag='loss95ci_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss95ci_meta_eval',
                scalar_value=loss_95ci, global_step=time_
            )
            writer.add_scalar(
                tag='acc95ci_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'acc95ci_meta_eval',
                scalar_value=acc_95ci, global_step=time_
            )
            print("ACC_VAL:{}".format(accuracy_eval))
        
        # update mean and covariance of meta-parameters
        var_beta.update_posterior(len(var_approx_list))
        for var_approx in var_approx_list:
            var_approx.update_mean_cov() 
        
        # save the previous beta distribution
        var_previous_beta_alpha.append(var_beta.var_gamma1.data.cpu().numpy())
        var_previous_beta_beta.append(var_beta.var_gamma2.data.cpu().numpy())

        # evidential sparsity
        if config['sparsity'] and len(var_approx_list) != 1:
            sparsity_vec = evidential_sparsity(var_beta,var_previous_beta_alpha, var_previous_beta_beta, config['eta'])
            delete_num = np.sum(np.where(sparsity_vec==1))

            # update the beta
            var_beta.delete_cluster(sparsity_vec)
            config['k'] -= delete_num

            # update the theta
            pop_index = np.where(sparsity_vec==0)[0]
            for i in range(1,len(pop_index)+1):
                var_approx_list.pop(pop_index[-i])

        torch.cuda.empty_cache()

def metatrain_seqdataset_multi_amvi(model, var_approx_list, var_beta, model_device, trainset, evalset, outer_kl_scale, 
        nstep_outer, nstep_inner, lr_inner, first_order, seqtask, num_task_per_itr, 
        task_by_supercls, num_way, num_shot, num_query_per_cls, eval_prev_task, eval_per_num_iter, 
        num_eval_task, eval_task_by_supercls, nstep_inner_eval, writer, task_idx, prev_glob_step, verbose, sample_mean,sample_batch, max_eval):
    '''
    train the model on a certain dataset and save some related losses
    input:
    output: None
    '''
    num_cluster = len(var_approx_list)
    
    # print(trainset.task_list)
    # print(trainset.df)
    # Done: prepara the data
    trainsampler = SuppQueryBatchSampler(
        dataset=trainset, seqtask=False, num_task=num_task_per_itr, task_by_supercls=task_by_supercls,
        num_way=num_way, num_shot=num_shot, num_query_per_cls=num_query_per_cls
    )
    # print(trainsampler.dataset.df)
    trainloader = DataLoader(trainset, batch_sampler=trainsampler)

    # Done: update loop, consisting training and evaluation
    for itr in trange(nstep_outer, desc='meta-train {}'.format(verbose if verbose is not None else task_idx), ncols=100):
        
        negloglik_supp, negloglik_query, loss_outer, accuracy_query = \
        outer_step(var_approx_list, model, var_beta, trainloader, num_way, num_shot, num_query_per_cls,\
                outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, model_device, sample_mean, sample_batch)
        
        writer.add_scalar(
                    tag='loss_meta_train_task{}'.format(task_idx),
                    scalar_value=loss_outer, global_step=prev_glob_step + itr
                )
        
        writer.add_scalar(
                    tag='accuracy_meta_train_task{}'.format(task_idx),
                    scalar_value=accuracy_query, global_step=prev_glob_step + itr
                )
        
        # var_beta.reset_prior()

        # Done: the evaluation process
        if (itr+1) % eval_per_num_iter == 0:
            
            # define the evaluation set at the current time
            if not eval_prev_task:
                evalset = [evalset[-1]]

            for ldr_idx, evset in enumerate(evalset):
                
                # loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi(
                # evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_way=num_way,
                # num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=model,
                # variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval,
                # lr_inner=lr_inner, model_device=model_device, var_beta=var_beta, sample_batch=sample_batch
                # )

                if not max_eval:
                    loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi_v2(
                    evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_way=num_way,
                    num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=model,
                    variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval,
                    lr_inner=lr_inner, model_device=model_device, var_beta=var_beta, sample_batch=sample_batch
                    )
                else:
                    loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi_v3(
                    evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_way=num_way,
                    num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=model,
                    variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval,
                    lr_inner=lr_inner, model_device=model_device, var_beta=var_beta, sample_batch=sample_batch
                    )

                # Done: save the related losses
                writer.add_scalar(
                    tag='loss_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss_meta_eval',
                    scalar_value=loss_eval, global_step=prev_glob_step + itr
                )
                writer.add_scalar(
                    tag='accuracy_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'accuracy_meta_eval',
                    scalar_value=accuracy_eval, global_step=prev_glob_step + itr
                )
                writer.add_scalar(
                    tag='loss95ci_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss95ci_meta_eval',
                    scalar_value=loss_95ci, global_step=prev_glob_step + itr
                )
                writer.add_scalar(
                    tag='acc95ci_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'acc95ci_meta_eval',
                    scalar_value=acc_95ci, global_step=prev_glob_step + itr
                )

    

def outer_step(var_approx_list, model, var_beta, dataloader, num_way, num_shot, num_query_per_cls, outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, model_device, sample_mean, sample_batch):
    '''
    outer update once, consisting of the parameter update and loss compute
    '''

    # Done: compute the loss and gradient using other functions
    grad_wrt_beta, grad_wrt_mean_all, grad_wrt_covar_all, negloglik_supp, negloglik_query, \
        loss_outer, acc_query = outer_gradient_amvi(var_approx_list, \
            model, var_beta, dataloader, num_way, num_shot, num_query_per_cls, outer_kl_scale, nstep_inner, \
                lr_inner, first_order, num_task_per_itr, sample_mean, sample_batch)

    # Done: compute the gradient of beta in the KL-terms, beta and z_k
    nll_querysupp_one_task_divisor = (num_shot + num_query_per_cls) * num_way * var_approx_list[0].num_mc_sample
    beta_kl_term = var_beta.KL_terms()/(nll_querysupp_one_task_divisor*num_task_per_itr)
    var_beta.optimizer.zero_grad()
    beta_kl_term.backward()
    beta_kl_gradient_wrt = OrderedDict([
        (name, parameter.grad) for (name, parameter) in var_beta.meta_named_parameters()
    ])

    with torch.no_grad():
        for grad_beta, grad_kl_beta in zip(grad_wrt_beta.values(),beta_kl_gradient_wrt.values()):
            grad_beta += grad_kl_beta
            grad_kl_beta.zero_()
        
        loss_outer += beta_kl_term

    # Done: update the variational distribution of beta
    total_grad_wrt_beta = [grad for grad in grad_wrt_beta.values()]
    var_beta.optimizer.step_grad(gradient=total_grad_wrt_beta)
    var_beta.check()

    # Done: update the variational distribution of theta
    for i, var_obj in enumerate(var_approx_list):
        total_grad_wrt_mean = [grad for grad in grad_wrt_mean_all[i].values()]
        total_grad_wrt_covar = [grad for grad in grad_wrt_covar_all[i].values()]
        var_obj.optimizer.step_grad(gradient=total_grad_wrt_mean + total_grad_wrt_covar)

    return negloglik_supp, negloglik_query, loss_outer, acc_query

def outer_step_seqtask(var_approx_list, model, var_beta, images, labels, num_way, num_shot, num_query_per_cls, outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, model_device, sample_mean, sample_batch, seq_task):
    '''
    outer update once, consisting of the parameter update and loss compute
    '''

    # Done: compute the loss and gradient using other functions
    grad_wrt_beta, grad_wrt_mean_all, grad_wrt_covar_all, negloglik_supp, negloglik_query, \
        loss_outer, acc_query = outer_gradient_amvi_seqtask(var_approx_list, \
            model, var_beta, images, labels, num_way, num_shot, num_query_per_cls, outer_kl_scale, nstep_inner, \
                lr_inner, first_order, num_task_per_itr, sample_mean, sample_batch)



    # Done: compute the gradient of beta in the KL-terms, beta and z_k
    nll_querysupp_one_task_divisor = (num_shot + num_query_per_cls) * num_way * var_approx_list[0].num_mc_sample
    beta_kl_term = var_beta.KL_terms()/(nll_querysupp_one_task_divisor*num_task_per_itr)
    var_beta.optimizer.zero_grad()
    beta_kl_term.backward()
    beta_kl_gradient_wrt = OrderedDict([
        (name, parameter.grad) for (name, parameter) in var_beta.meta_named_parameters()
    ])

    with torch.no_grad():
        for grad_beta, grad_kl_beta in zip(grad_wrt_beta.values(),beta_kl_gradient_wrt.values()):
            grad_beta += grad_kl_beta
            grad_kl_beta.zero_()
        
        loss_outer += beta_kl_term

    # Done: update the variational distribution of beta
    total_grad_wrt_beta = [grad for grad in grad_wrt_beta.values()]
    var_beta.optimizer.step_grad(gradient=total_grad_wrt_beta)
    var_beta.check()

    # Done: update the variational distribution of theta
    for i, var_obj in enumerate(var_approx_list):
        total_grad_wrt_mean = [grad for grad in grad_wrt_mean_all[i].values()]
        total_grad_wrt_covar = [grad for grad in grad_wrt_covar_all[i].values()]
        var_obj.optimizer.step_grad(gradient=total_grad_wrt_mean + total_grad_wrt_covar)

    return negloglik_supp, negloglik_query, loss_outer, acc_query


def outer_gradient_amvi(var_approx_list, model, var_beta, dataloader, num_way, num_shot, num_query_per_cls, 
                        outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, sample_mean, sample_batch):

    # Done: create some variable to accumulate the gradient and loss
    # for gradient accumulation
    grad_wrt_mean_all = []
    grad_wrt_covar_all = []

    # initialize the gradident
    grad_wrt_beta_all = OrderedDict([
            (name, torch.zeros_like(parameter)) for name, parameter in var_beta.meta_named_parameters()
        ])

    for var_obj in var_approx_list:
        grad_wrt_mean_all.append(OrderedDict([
            (name, torch.zeros_like(mu)) for name, mu in var_obj.mean.items()
        ]))
        grad_wrt_covar_all.append(OrderedDict([
            (name, torch.zeros_like(cov)) for name, cov in var_obj.covar.items()
        ]))
    
    # initial the loss
    negloglik_query = torch.tensor(0., device=var_beta.device)
    negloglik_supp = torch.tensor(0., device=var_beta.device)
    loss_outer_all = torch.tensor(0., device=var_beta.device)
    accuracy_query = torch.tensor(0., device=var_beta.device)
    nll_querysupp_one_task_divisor = (num_shot + num_query_per_cls) * num_way * var_approx_list[0].num_mc_sample # the number pf sample

    # Done: train for batch
    for batch_idx, (images, labels) in enumerate(dataloader, 0):
        
        # compute the gradient for each batch
        grad_wrt_beta, grad_wrt_mean_list, grad_wrt_covar_list, nll_supp, nll_query, loss_outer, acc_query = \
            batch_gradient_amvi(var_approx_list, var_beta, model, images, labels, num_way, num_shot, num_query_per_cls,
            nstep_inner, lr_inner, first_order, num_task_per_itr, sample_mean, sample_batch)
            
        # accumulate the loss
        negloglik_supp += nll_supp
        negloglik_query += nll_query
        loss_outer_all += loss_outer
        accuracy_query += acc_query

        # accumulate the gradient
        for i, var_obj in enumerate(var_approx_list):

            for acc_g_mu, acc_g_cov, g_mu, g_cov in \
                zip(grad_wrt_mean_all[i].values(), grad_wrt_covar_all[i].values(),
                    grad_wrt_mean_list[i].values(), grad_wrt_covar_list[i].values()):
                        
                acc_g_mu += g_mu
                acc_g_cov += g_cov
                # zero gradients after accumulating mean and cov gradients
                g_mu.zero_()
                g_cov.zero_()

        # accumulate the gradient of beta
        for acc_g_beta, g_beta in \
            zip(grad_wrt_beta_all.values(), grad_wrt_beta.values()):
                    
            acc_g_beta += g_beta
            # zero gradients after accumulating mean and cov gradients
            g_beta.zero_()


    # Done: Compute the gradient on the KL-term of theta
    for i in range(len(var_approx_list)):
        var_obj = var_approx_list[i]
        kldiv = outer_kl_scale * kldiv_mvn_diagcov(
            mean_p=var_obj.mean, cov_p=exp_covar(var_obj.covar),
            mean_q=var_obj.mean_old, cov_q=exp_covar(var_obj.covar_old)
        ) / (nll_querysupp_one_task_divisor * num_task_per_itr)

        # add the kl loss to the outer loss
        with torch.no_grad():
            loss_outer += kldiv
        
        var_obj.optimizer.zero_grad()
        kldiv.backward()
        kldiv_gradient_wrt_mean = OrderedDict([(name, mu.grad) for name, mu in var_obj.mean.items()])
        kldiv_gradient_wrt_covar = OrderedDict([(name, cov.grad) for name, cov in var_obj.covar.items()])

        # Done: accumulate the gradient of each loss and return
        with torch.no_grad():
            for acc_g_mu, acc_g_cov, kldiv_g_mu, kldiv_g_cov in \
                zip(grad_wrt_mean_all[i].values(), grad_wrt_covar_all[i].values(), \
                    kldiv_gradient_wrt_mean.values(), kldiv_gradient_wrt_covar.values()):
                acc_g_mu += kldiv_g_mu
                acc_g_cov += kldiv_g_cov
                kldiv_g_mu.zero_()
                kldiv_g_cov.zero_()

    # compute the average loss and accuracy
    negloglik_supp /= num_task_per_itr
    negloglik_query /= num_task_per_itr
    loss_outer_all /= num_task_per_itr
    accuracy_query /= num_task_per_itr

    return grad_wrt_beta_all, grad_wrt_mean_all, grad_wrt_covar_all, negloglik_supp, negloglik_query, \
            loss_outer, acc_query 

def outer_gradient_amvi_seqtask(var_approx_list, model, var_beta, images, labels, num_way, num_shot, num_query_per_cls, 
                        outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, sample_mean, sample_batch):

    # Done: create some variable to accumulate the gradient and loss
    # for gradient accumulation
    grad_wrt_mean_all = []
    grad_wrt_covar_all = []

    # initialize the gradident
    grad_wrt_beta_all = OrderedDict([
            (name, torch.zeros_like(parameter)) for name, parameter in var_beta.meta_named_parameters()
        ])

    for var_obj in var_approx_list:
        grad_wrt_mean_all.append(OrderedDict([
            (name, torch.zeros_like(mu)) for name, mu in var_obj.mean.items()
        ]))
        grad_wrt_covar_all.append(OrderedDict([
            (name, torch.zeros_like(cov)) for name, cov in var_obj.covar.items()
        ]))
    
    # initial the loss
    negloglik_query = torch.tensor(0., device=var_beta.device)
    negloglik_supp = torch.tensor(0., device=var_beta.device)
    loss_outer_all = torch.tensor(0., device=var_beta.device)
    accuracy_query = torch.tensor(0., device=var_beta.device)
    nll_querysupp_one_task_divisor = (num_shot + num_query_per_cls) * num_way * var_approx_list[0].num_mc_sample # the number pf sample
        
    # compute the gradient for each batch
    grad_wrt_beta, grad_wrt_mean_list, grad_wrt_covar_list, nll_supp, nll_query, loss_outer, acc_query = \
        batch_gradient_amvi(var_approx_list, var_beta, model, images, labels, num_way, num_shot, num_query_per_cls,
        nstep_inner, lr_inner, first_order, num_task_per_itr, sample_mean, sample_batch)
        
    # accumulate the loss
    negloglik_supp += nll_supp
    negloglik_query += nll_query
    loss_outer_all += loss_outer
    accuracy_query += acc_query

    # accumulate the gradient
    for i, var_obj in enumerate(var_approx_list):

        for acc_g_mu, acc_g_cov, g_mu, g_cov in \
            zip(grad_wrt_mean_all[i].values(), grad_wrt_covar_all[i].values(),
                grad_wrt_mean_list[i].values(), grad_wrt_covar_list[i].values()):
                    
            acc_g_mu += g_mu
            acc_g_cov += g_cov
            # zero gradients after accumulating mean and cov gradients
            g_mu.zero_()
            g_cov.zero_()

    # accumulate the gradient of beta
    for acc_g_beta, g_beta in \
        zip(grad_wrt_beta_all.values(), grad_wrt_beta.values()):
                
        acc_g_beta += g_beta
        # zero gradients after accumulating mean and cov gradients
        g_beta.zero_()

    # Done: Compute the gradient on the KL-term of theta
    for i in range(len(var_approx_list)):
        var_obj = var_approx_list[i]
        kldiv = outer_kl_scale * kldiv_mvn_diagcov(
            mean_p=var_obj.mean, cov_p=exp_covar(var_obj.covar),
            mean_q=var_obj.mean_old, cov_q=exp_covar(var_obj.covar_old)
        ) / (nll_querysupp_one_task_divisor * num_task_per_itr)

        # add the kl loss to the outer loss
        with torch.no_grad():
            loss_outer += kldiv
        
        var_obj.optimizer.zero_grad()
        kldiv.backward()
        kldiv_gradient_wrt_mean = OrderedDict([(name, mu.grad) for name, mu in var_obj.mean.items()])
        kldiv_gradient_wrt_covar = OrderedDict([(name, cov.grad) for name, cov in var_obj.covar.items()])

        # Done: accumulate the gradient of each loss and return
        with torch.no_grad():
            for acc_g_mu, acc_g_cov, kldiv_g_mu, kldiv_g_cov in \
                zip(grad_wrt_mean_all[i].values(), grad_wrt_covar_all[i].values(), \
                    kldiv_gradient_wrt_mean.values(), kldiv_gradient_wrt_covar.values()):
                acc_g_mu += kldiv_g_mu
                acc_g_cov += kldiv_g_cov
                kldiv_g_mu.zero_()
                kldiv_g_cov.zero_()

    # compute the average loss and accuracy
    negloglik_supp /= num_task_per_itr
    negloglik_query /= num_task_per_itr
    loss_outer_all /= num_task_per_itr
    accuracy_query /= num_task_per_itr

    return grad_wrt_beta_all, grad_wrt_mean_all, grad_wrt_covar_all, negloglik_supp, negloglik_query, \
            loss_outer, acc_query 


def batch_gradient_amvi(var_approx_list, var_beta, model, images, labels, num_way, num_shot, num_query_per_cls,
                        nstep_inner, lr_inner, first_order, num_batch, sample_mean, sample_batch):
    '''
    input:
    output:
        grad_wrt_beta: the gradient of beta on the neglogloss
        grad_wrt_mean_list: the gradient of mean on the neglog
        grad_wrt_covar_list: the gradient of covar on the neglog
        negloglik_supp
        negloglik_query
        accuracy_query
    '''

    # Done: divide the dataset to support set and query set
    device = var_beta.device

    # init the gradient
    grad_wrt_mean_list = []
    grad_wrt_covar_list = []

    grad_wrt_beta_all = OrderedDict([
            (name, torch.zeros_like(parameter)) for name, parameter in var_beta.meta_named_parameters()
        ])

    for var_obj in var_approx_list:
        grad_wrt_mean_list.append(OrderedDict([
            (name, torch.zeros_like(mu)) for name, mu in var_obj.mean.items()
        ]))
        grad_wrt_covar_list.append(OrderedDict([
            (name, torch.zeros_like(cov)) for name, cov in var_obj.covar.items()
        ]))
    
    # define the support and quert set
    supp_idx = num_way * num_shot
    ims = images.to(device=device)
    lbls = labels.to(device=device)
    support_img, query_img = ims[:supp_idx, :], ims[supp_idx:, :]
    support_lbl, query_lbl = lbls[:supp_idx], lbls[supp_idx:]

    # init the loss
    negloglik_query_batch = torch.tensor(0., device=var_beta.device)
    negloglik_supp_batch = torch.tensor(0., device=var_beta.device)
    loss_outer_batch = torch.tensor(0., device=var_beta.device)
    accuracy_query_batch = torch.tensor(0., device=var_beta.device)
    nll_querysupp_one_task_divisor = (num_shot + num_query_per_cls) * num_way * var_approx_list[0].num_mc_sample # the number pf sample

    sample_num = 0
    if sample_mean:
        sample_num = var_beta.num_mc_sample
        var_beta.num_mc_sample = 1

    for _ in range(var_beta.num_mc_sample):

        # Done: sample for z
        # beta_sample =  var_beta.sample()
        if sample_mean:
            pis = torch.zeros_like(var_beta.var_gamma1).to(var_beta.device)
            for _ in range(sample_num):
                try:
                    beta_sample =  var_beta.sample()
                except ValueError:
                    print(var_beta.var_gamma1)
                    print(var_beta.var_gamma2)
                    print(var_beta.bern_pro)
                    print(aaa)
                pis += beta_sample
            
            
            pis /= sample_num
            
            # pos_pis = var_beta.bern_pro + pis
            z_t = reparam_bernoulli(pis)
        else:
            beta_sample =  var_beta.sample()
            pis = beta_sample
            # pos_pis = var_beta.bern_pro + pis
            z_t = reparam_bernoulli(pis)
        
        # Done: compute the related loss of batch, including neglogsupp, neglogquery and accuracy
        # negloglik_supp, negloglik_query, accuracy_query = \
        #     batch_loss_multi_amvi(support_img, support_lbl, query_img, query_lbl, var_beta,
        #                     var_approx_list, z_t, model, nstep_inner, lr_inner, first_order, device, 
        #                     cal_zk_likelihood=var_beta is not None, 
        #                     sample_batch=sample_batch)

        # negloglik_supp, negloglik_query, accuracy_query = \
        #     batch_loss_multi_amvi(support_img, support_lbl, query_img, query_lbl, var_beta,
        #                     var_approx_list, pos_pis, model, nstep_inner, lr_inner, first_order, device, 
        #                     cal_zk_likelihood=var_beta is not None, 
        #                     sample_batch=sample_batch)

        negloglik_supp, negloglik_query, accuracy_query = \
            batch_loss_multi_amvi_v2(support_img, support_lbl, query_img, query_lbl, var_beta,
                            var_approx_list, pis, model, nstep_inner, num_way, lr_inner, first_order, device, 
                            cal_zk_likelihood=var_beta is not None, 
                            sample_batch=sample_batch)


        nll_querysupp = negloglik_query + negloglik_supp

        # Done: compute the bernouli kl
        # prior_sample_pis = var_beta.sample_prior_pros()
        # kl_bernouli_loss = kl_bernoulli(pis, z_t, pis) /(nll_querysupp_one_task_divisor)

        # all_loss = nll_querysupp + kl_bernouli_loss
        all_loss = nll_querysupp

        # zero the gradient
        for var_obj in var_approx_list:
            var_obj.optimizer.zero_grad()
        
        var_beta.optimizer.zero_grad()

        # backward the loss
        all_loss.backward()

        # accumulate the gradient of theta
        for i, var_obj in enumerate(var_approx_list):
            grad_wrt_mean = OrderedDict([
                (name, param.grad) for name, param in var_obj.mean.items()
            ])

            grad_wrt_covar = OrderedDict([
                (name, param.grad) for (name, param) in var_obj.covar.items()
            ])

            for acc_g_mu, acc_g_cov, g_mu, g_cov in \
                zip(grad_wrt_mean_list[i].values(), grad_wrt_covar_list[i].values(),
                    grad_wrt_mean.values(), grad_wrt_covar.values()):
                
                acc_g_mu += g_mu
                acc_g_cov += g_cov
                # zero gradients after accumulating mean and cov gradients
                g_mu.zero_()
                g_cov.zero_()

        # Done: compute the gradient of the beta distribution
        grad_wrt_beta = OrderedDict([
                (name, param.grad) for name, param in var_beta.meta_named_parameters()
            ])

        for acc_g_beta, g_beta in \
            zip(grad_wrt_beta_all.values(), grad_wrt_beta.values()):
    
            acc_g_beta += g_beta

            # zero gradients after accumulating mean and cov gradients
            g_beta.zero_()
        

        # accumulate the loss
        negloglik_query_batch += negloglik_query
        negloglik_supp_batch += negloglik_supp
        loss_outer_batch += (negloglik_query+negloglik_supp)
        accuracy_query_batch += accuracy_query

    # compute the average loss
    negloglik_supp_batch /= var_beta.num_mc_sample
    negloglik_query_batch /= var_beta.num_mc_sample
    loss_outer_batch /= var_beta.num_mc_sample
    accuracy_query_batch /= var_beta.num_mc_sample

    return grad_wrt_beta_all, grad_wrt_mean_list, grad_wrt_covar_list, negloglik_supp_batch, negloglik_query_batch, loss_outer_batch, accuracy_query_batch


def batch_loss_multi_amvi(support_img, support_lbl, query_img, query_lbl, var_beta, var_obj_list, z_t, model, 
                    nstep_inner, lr_inner, first_order, device,  cal_zk_likelihood=False, 
                    sample_batch=True):
    """
    input:
        dataset: consisting of support set and query set
        var_obj_list: all clusters of theta
        moodel, nstep_inner, lr_inner, first_order, device
        cal_zk_likehood: if computing the likelihood of z_k
        var_beta: the variational distribution of beta
    output: 
        nll_supp: the loss on the support set
        nll_query: the loss on the query set
        accuracy_query: the accuracy on query
    """

    # Done: to compute the specific parameter, that is, the inner update
    if not sample_batch:
        support_img = support_img.expand(1, *support_img.size())
        support_lbl = support_lbl.expand(1, *support_lbl.size())
        query_img = query_img.expand(1, *query_img.size())
        query_lbl = query_lbl.expand(1, *query_lbl.size())

        mean_inner, cov_inner = inner_maml_amvi(
            model=model, var_obj_list=var_obj_list, z_t = z_t, inputs=support_img, labels=support_lbl, 
            nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=first_order, 
            device=device, kl_scale=0.01, sample_batch=False
        )

        nll_supp = torch.tensor(0.0, device=device)
        nll_query = torch.tensor(0.0, device=device)
        acc_query = torch.tensor(0.0, device=device)

        # Done: sampling the final result to compute the negative log likelihood
        for _ in range(var_obj_list[0].num_mc_sample):
            # MC Sample for likehood 
            out_supp = model(x=support_img, mean=mean_inner, cov=var_obj_list[0].exp_covar(cov_inner))
            nll_supp += cross_entropy(input=out_supp, target=support_lbl, reduction='mean')
            out_query = model(x=query_img, mean=mean_inner, cov=var_obj_list[0].exp_covar(cov_inner))
            nll_query += cross_entropy(input=out_query, target=query_lbl, reduction='mean')

            with torch.no_grad():
                acc_query += get_accuracy(labels=query_lbl, outputs=out_query)
        
        # compute the average of each sample
        nll_supp /= var_obj_list[0].num_mc_sample
        nll_query /= var_obj_list[0].num_mc_sample
        acc_query /= var_obj_list[0].num_mc_sample


        return nll_supp, nll_query, acc_query
    
    else:
        support_img = support_img.expand(var_obj_list[0].num_mc_sample, *support_img.size())
        support_lbl = support_lbl.expand(var_obj_list[0].num_mc_sample, *support_lbl.size())
        query_img = query_img.expand(var_obj_list[0].num_mc_sample, *query_img.size())
        query_lbl = query_lbl.expand(var_obj_list[0].num_mc_sample, *query_lbl.size())

        # inner update to get the specific model parameter
        mean_inner, cov_inner = inner_maml_amvi(
            model=model, var_obj_list=var_obj_list, z_t = z_t, inputs=support_img, labels=support_lbl, 
            nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=first_order, 
            device=device, kl_scale=0.01, sample_batch=True
        )

        # compute the loss on the support set and query set
        out_supp = model(x=support_img, mean=mean_inner, cov=exp_covar(cov_inner))
        nll_supp = cross_entropy(input=out_supp, target=support_lbl, reduction='mean')
        out_query = model(x=query_img, mean=mean_inner, cov=exp_covar(cov_inner))
        nll_query = cross_entropy(input=out_query, target=query_lbl, reduction='mean')
        with torch.no_grad():
            accuracy_query = get_accuracy(labels=query_lbl, outputs=out_query)

        return nll_supp, nll_query, accuracy_query


def batch_loss_multi_amvi_v2(support_img, support_lbl, query_img, query_lbl, var_beta, var_obj_list, z_t, model, 
                nstep_inner, num_way, lr_inner, first_order, device,  cal_zk_likelihood=False, 
                sample_batch=True):
    
    if not sample_batch:
        pass
    else:
        support_img = support_img.expand(var_obj_list[0].num_mc_sample, *support_img.size())
        support_lbl = support_lbl.expand(var_obj_list[0].num_mc_sample, *support_lbl.size())
        query_img = query_img.expand(var_obj_list[0].num_mc_sample, *query_img.size())
        query_lbl = query_lbl.expand(var_obj_list[0].num_mc_sample, *query_lbl.size())
        

        supp_mean_list = []
        supp_covar_list = []
        query_mean_list = []
        query_covar_list = []

        # compute the prediction result
        for var_obj in var_obj_list:

            mean_inner, covar_inner = inner_maml_amvi_v2(
                model=model, var_obj=var_obj, inputs=support_img, labels=support_lbl, 
                nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=first_order, 
                device=device, kl_scale=0.01, sample_batch=True
            )

            supp_predict_mean, supp_predict_covar = model(x=support_img, mean=mean_inner, cov=exp_covar(covar_inner), output_type=False)
            supp_mean_list.append(supp_predict_mean)
            supp_covar_list.append(supp_predict_covar)
            #nll_supp = cross_entropy(input=out_supp, target=support_lbl, reduction='mean')
            query_predict_mean, query_predict_covar = model(x=query_img, mean=mean_inner, cov=exp_covar(covar_inner), output_type=False)
            query_mean_list.append(query_predict_mean)
            query_covar_list.append(query_predict_covar)
            #nll_query = cross_entropy(input=out_query, target=query_lbl, reduction='mean')

        supp_rand = torch.rand_like(supp_mean_list[0])
        query_rand = torch.rand_like(query_covar_list[0])
        out_supp = torch.zeros([support_img.size()[0], support_img.size()[1], num_way], device=var_beta.device)
        out_query = torch.zeros([query_img.size()[0], query_img.size()[1], num_way], device = var_beta.device)
        
        for i in range(len(supp_mean_list)):

            # for support set
            supp_predict = (supp_mean_list[i]+supp_covar_list[i]*supp_rand)
            supp_predict = supp_predict.reshape(support_img.size()[0], support_img.size()[1], *supp_predict.size()[1:])
            out_supp += z_t[i] * (supp_predict)
            
        
            # for query set
            query_predict = (query_mean_list[i]+query_covar_list[i]*query_rand)
            query_predict = query_predict.reshape(query_img.size()[0], query_img.size()[1], *query_predict.size()[1:])
            out_query += z_t[i] * (query_predict)
        
        nll_supp = cross_entropy(input=out_supp, target=support_lbl, reduction='mean')
        nll_query = cross_entropy(input=out_query, target=query_lbl, reduction='mean')

        with torch.no_grad():
            accuracy_query = get_accuracy(labels=query_lbl, outputs=out_query)

        return nll_supp, nll_query, accuracy_query