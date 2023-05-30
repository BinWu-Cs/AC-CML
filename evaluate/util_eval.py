import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from collections import OrderedDict
import sys
sys.path.append(".")
from data_generate.sampler import SuppQueryBatchSampler
from functional.cross_entropy import cross_entropy
from train.inner import inner_maml_amvi, inner_maml_amvi_v2
from train.util import get_accuracy, get_zk_likelihood, exp_covar, reparam_bernoulli

def meta_evaluation_amvi(evalset, num_task, task_by_supercls, num_way, num_shot, num_query_per_cls, model, 
                        variational_obj_list, inner_on_mean, n_sample=1, nstep_inner=10, lr_inner=0.4, 
                        model_device=None, var_beta=None, sample_batch=True, return_eta=False):
    
    # define the variable to save the loss and accuracy
    loss = []
    accuracy = []

    # define the sample for evaluation
    evalsampler = SuppQueryBatchSampler(
        dataset=evalset, seqtask=False, num_task=num_task, task_by_supercls=task_by_supercls, num_way=num_way,
        num_shot=num_shot, num_query_per_cls=num_query_per_cls
    )
    evalloader = DataLoader(evalset, batch_sampler=evalsampler)

    # loop for evaluation
    for images, labels in evalloader:

        # load the data into the device
        device = var_beta.device
        expand_dim = variational_obj_list[0].num_mc_sample if sample_batch else 1
        ims = images.to(device=device)
        lbls = labels.to(device=device)

        # init the loss and acc
        nll_query = torch.tensor(0., device=device)
        acc_query = torch.tensor(0., device=device)

        # divide the dataset to support set and query set
        supp_idx = num_way * num_shot
        support_img, query_img = ims[:supp_idx, :], ims[supp_idx:, :]
        support_lbl, query_lbl = lbls[:supp_idx], lbls[supp_idx:]

        support_img = support_img.expand(expand_dim, *support_img.size())
        support_lbl = support_lbl.expand(expand_dim, *support_lbl.size())
        query_img = query_img.expand(expand_dim, *query_img.size())
        query_lbl = query_lbl.expand(expand_dim, *query_lbl.size())

        # sample the z and compute the inner parameter
        beta_sample =  var_beta.sample()
        pis = torch.cumprod(beta_sample, dim=0)
        # z_t = reparam_bernoulli(pis)
        z_t = pis
        
        mean_inner, cov_inner = inner_maml_amvi(
                model=model, var_obj_list=variational_obj_list, z_t=z_t, inputs=support_img, labels=support_lbl, 
                nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=True, device=device,
                kl_scale=0.01, sample_batch=sample_batch
            )


        if sample_batch:
            
            output = model(x=query_img, mean=mean_inner, cov=exp_covar(cov_inner))
        
            with torch.no_grad():
                nll_query = cross_entropy(input=output, target=query_lbl, reduction='mean') / variational_obj_list[0].num_mc_sample
                acc_query = get_accuracy(labels=query_lbl, outputs=output)
        
        else:
            output = model(x=query_img, mean=mean_inner, cov=exp_covar(cov_inner))
            with torch.no_grad():
                nll_query = cross_entropy(input=output, target=query_lbl, reduction='mean') / variational_obj_list[0].num_mc_sample
                acc_query = get_accuracy(labels=query_lbl, outputs=output) / variational_obj_list[0].num_mc_sample

        loss.append(nll_query)
        accuracy.append(acc_query)
    
    # compute the average result
    loss_tensor = torch.stack(loss)
    acc_tensor = torch.stack(accuracy)

    loss_mean = loss_tensor.mean()
    acc_mean = acc_tensor.mean()

    sqrt_nsample = (torch.tensor(num_task, dtype=torch.float32, device=model_device)).sqrt()
    loss_95ci = 1.96 * loss_tensor.std(unbiased=True) / sqrt_nsample
    acc_95ci = 1.96 * acc_tensor.std(unbiased=True) / sqrt_nsample

    return loss_mean.item(), loss_95ci.item(), acc_mean.item(), acc_95ci.item()
    

def meta_evaluation_amvi_v2(evalset, num_task, task_by_supercls, num_way, num_shot, num_query_per_cls, model, 
                        variational_obj_list, inner_on_mean, n_sample=1, nstep_inner=10, lr_inner=0.4, 
                        model_device=None, var_beta=None, sample_batch=True, return_eta=False):
    
    # define the variable to save the loss and accuracy
    loss = []
    accuracy = []

    # define the sample for evaluation
    evalsampler = SuppQueryBatchSampler(
        dataset=evalset, seqtask=False, num_task=num_task, task_by_supercls=task_by_supercls, num_way=num_way,
        num_shot=num_shot, num_query_per_cls=num_query_per_cls
    )
    evalloader = DataLoader(evalset, batch_sampler=evalsampler)

    # loop for evaluation
    for images, labels in evalloader:

        # load the data into the device
        device = var_beta.device
        expand_dim = variational_obj_list[0].num_mc_sample if sample_batch else 1
        ims = images.to(device=device)
        lbls = labels.to(device=device)

        # init the loss and acc
        nll_query = torch.tensor(0., device=device)
        acc_query = torch.tensor(0., device=device)

        # divide the dataset to support set and query set
        supp_idx = num_way * num_shot
        support_img, query_img = ims[:supp_idx, :], ims[supp_idx:, :]
        support_lbl, query_lbl = lbls[:supp_idx], lbls[supp_idx:]

        support_img = support_img.expand(expand_dim, *support_img.size())
        support_lbl = support_lbl.expand(expand_dim, *support_lbl.size())
        query_img = query_img.expand(expand_dim, *query_img.size())
        query_lbl = query_lbl.expand(expand_dim, *query_lbl.size())

        # sample the z and compute the inner parameter
        beta_sample =  var_beta.sample()
        pis = torch.cumprod(beta_sample, dim=0)
        # z_t = reparam_bernoulli(pis)
        z_t = pis
        
        result_mean_list = []
        result_covar_list = []

        for var_obj in variational_obj_list:

            mean_inner, covar_inner = inner_maml_amvi_v2(
                model=model, var_obj=var_obj, inputs=support_img, labels=support_lbl, 
                nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=True, 
                device=device, kl_scale=0.01, sample_batch=True
            )

            result_mean, result_covar = model(x=query_img, mean=mean_inner, cov=exp_covar(covar_inner), output_type=False)
            result_mean_list.append(result_mean)
            result_covar_list.append(result_covar)


        if sample_batch:
            
            result_rand = torch.rand_like(result_mean_list[0])
            out_result = torch.zeros([query_img.size()[0], query_img.size()[1], num_way], device = var_beta.device)

            for i in range(len(result_mean_list)):

                result_predict = (result_mean_list[i]+result_covar_list[i]*result_rand)
                result_predict = result_predict.reshape(query_img.size()[0], query_img.size()[1], *result_predict.size()[1:])
                out_result += z_t[i] * (result_predict)
            
            with torch.no_grad():
                nll_query = cross_entropy(input=out_result, target=query_lbl, reduction='mean') / variational_obj_list[0].num_mc_sample
                acc_query = get_accuracy(labels=query_lbl, outputs=out_result)

        else:
            pass

        loss.append(nll_query)
        accuracy.append(acc_query)
    
    # compute the average result
    loss_tensor = torch.stack(loss)
    acc_tensor = torch.stack(accuracy)

    loss_mean = loss_tensor.mean()
    acc_mean = acc_tensor.mean()

    sqrt_nsample = (torch.tensor(num_task, dtype=torch.float32, device=model_device)).sqrt()
    loss_95ci = 1.96 * loss_tensor.std(unbiased=True) / sqrt_nsample
    acc_95ci = 1.96 * acc_tensor.std(unbiased=True) / sqrt_nsample

    return loss_mean.item(), loss_95ci.item(), acc_mean.item(), acc_95ci.item()


def meta_evaluation_amvi_v3(evalset, num_task, task_by_supercls, num_way, num_shot, num_query_per_cls, model, 
                        variational_obj_list, inner_on_mean, n_sample=1, nstep_inner=10, lr_inner=0.4, 
                        model_device=None, var_beta=None, sample_batch=True, return_eta=False, max_index=3):
    
    # define the variable to save the loss and accuracy
    loss = []
    accuracy = []

    # define the sample for evaluation
    evalsampler = SuppQueryBatchSampler(
        dataset=evalset, seqtask=False, num_task=num_task, task_by_supercls=task_by_supercls, num_way=num_way,
        num_shot=num_shot, num_query_per_cls=num_query_per_cls
    )
    evalloader = DataLoader(evalset, batch_sampler=evalsampler)

    # loop for evaluation
    for images, labels in evalloader:
        # one task each loop

        # load the data into the device
        device = var_beta.device
        expand_dim = variational_obj_list[0].num_mc_sample if sample_batch else 1
        ims = images.to(device=device)
        lbls = labels.to(device=device)

        # init the loss and acc
        nll_query = torch.tensor(0., device=device)
        acc_query = torch.tensor(0., device=device)

        # divide the dataset to support set and query set
        supp_idx = num_way * num_shot
        support_img, query_img = ims[:supp_idx, :], ims[supp_idx:, :]
        support_lbl, query_lbl = lbls[:supp_idx], lbls[supp_idx:]

        support_img = support_img.expand(expand_dim, *support_img.size())
        support_lbl = support_lbl.expand(expand_dim, *support_lbl.size())
        query_img = query_img.expand(expand_dim, *query_img.size())
        query_lbl = query_lbl.expand(expand_dim, *query_lbl.size())
        
        result_mean_list = []
        result_covar_list = []
        supp_loss = []

        for var_obj in variational_obj_list:

            mean_inner, covar_inner = inner_maml_amvi_v2(
                model=model, var_obj=var_obj, inputs=support_img, labels=support_lbl, 
                nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=True, 
                device=device, kl_scale=0.01, sample_batch=True
            )

            result_mean, result_covar = model(x=query_img, mean=mean_inner, cov=exp_covar(covar_inner), output_type=False)
            result_mean_list.append(result_mean)
            result_covar_list.append(result_covar)

            if sample_batch:
            
                supp_output = model(x=support_img, mean=mean_inner, cov=exp_covar(covar_inner))
        
                with torch.no_grad():
                    nll_support = cross_entropy(input=supp_output, target=support_lbl, reduction='mean') / variational_obj_list[0].num_mc_sample

            else:
                output = model(x=query_img, mean=mean_inner, cov=exp_covar(covar_inner))
                with torch.no_grad():
                    nll_support = cross_entropy(input=supp_output, target=support_lbl, reduction='mean') / variational_obj_list[0].num_mc_sample
            
            supp_loss.append(nll_support.cpu())
        
        supp_loss_np = np.array(supp_loss)
        sample_index = supp_loss_np.argsort()[:max_index]

        if sample_batch:
            
            result_rand = torch.rand_like(result_mean_list[0])
            out_result = torch.zeros([query_img.size()[0], query_img.size()[1], num_way], device = var_beta.device)

            for i in sample_index:

                result_predict = (result_mean_list[i]+result_covar_list[i]*result_rand)
                result_predict = result_predict.reshape(query_img.size()[0], query_img.size()[1], *result_predict.size()[1:])
                out_result += result_predict
            
            with torch.no_grad():
                nll_query = cross_entropy(input=out_result, target=query_lbl, reduction='mean') / variational_obj_list[0].num_mc_sample
                acc_query = get_accuracy(labels=query_lbl, outputs=out_result)

        else:
            pass

        loss.append(nll_query)
        accuracy.append(acc_query)
    
    # compute the average result
    loss_tensor = torch.stack(loss)
    acc_tensor = torch.stack(accuracy)

    loss_mean = loss_tensor.mean()
    acc_mean = acc_tensor.mean()

    sqrt_nsample = (torch.tensor(num_task, dtype=torch.float32, device=model_device)).sqrt()
    loss_95ci = 1.96 * loss_tensor.std(unbiased=True) / sqrt_nsample
    acc_95ci = 1.96 * acc_tensor.std(unbiased=True) / sqrt_nsample

    return loss_mean.item(), loss_95ci.item(), acc_mean.item(), acc_95ci.item()