import shutil
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import datetime
import os
import json
import random
from platform import system
from warnings import filterwarnings
import sys
sys.path.append("..")

import optim
from config.configuration import get_run_name
from data_generate.dataset import FewShotImageDataset
from data_generate.sampler import SuppQueryBatchSampler
from train import model as models
from train.variational import  VariationalApprox_v2, var_approx_beta
from train.boml import metatrain_seqdataset_multi_amvi
from train.util import enlist_transformation
from train.evidential_sparsity import evidential_sparsity

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(config, run_spec, seed=0):
    
    torch.manual_seed(seed)

    start_datetime = datetime.datetime.now()
    experiment_date = '{:%Y-%m-%d_%H:%M:%S}'.format(start_datetime)
    config['experiment_parent_dir'] = os.path.join(config['run_dir'], get_run_name(config['dataset_ls']))
    config['experiment_dir'] = os.path.join(config['experiment_parent_dir'],
                                             '{}_{}_{}'.format(run_spec, experiment_date, seed))
    config['experiment_dir'] = os.path.join(config['experiment_parent_dir'],
                                            '{}'.format(run_spec+experiment_date))


    # print the related information on the screen
    os.system("echo 'running {}_{} seed {}'".format(run_spec, experiment_date, seed)) if system() == 'Linux' \
        else print('running {}_{} seed {}'.format(run_spec, experiment_date, seed))
    
    # save config json file
    # if not os.path.exists(config['experiment_dir']):
    #     os.makedirs(config['experiment_dir'])

    if os.path.exists(config['experiment_dir']):
        if (input('{} exists, remove? ([y]/n): '.format(config['experiment_dir'])) != 'n'):
            shutil.rmtree(config['experiment_dir'])
            os.makedirs(config['experiment_dir'])
        else:
            config['experiment_dir'] += experiment_date
            os.makedirs(config['experiment_dir'])
    else:
        os.makedirs(config['experiment_dir'])


    with open(os.path.join(
            config['experiment_dir'],
            'config{}_{}.json'.format(0 if config['completed_task_idx'] is None
                                      else config['completed_task_idx'] + 1, run_spec)
    ), 'w') as outfile:
        outfile.write(json.dumps(config, indent=4))

    # define result directory and previous result directory if applicable
    if config['completed_task_idx'] is not None:
        completed_result_dir = os.path.join(
            os.path.join(os.path.join(config['run_dir'], get_run_name(config['dataset_ls'])),
                         config['completed_exp_name']),
            'result'
        )
    else:
        completed_result_dir = None

    # define tensorboard writer
    writer = SummaryWriter(os.path.join(config['experiment_dir'], 'logtb'))
    result_dir = os.path.join(config['experiment_dir'], 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # define model
    '''
    Note that the initial variational object should be modified.
    '''
    model = getattr(models, config['net'])(**config['net_kwargs']).to(device=config['device'])
    
    # define the initial variational object of theta
    var_approx_list = []

    # define the list to save the previous veta
    var_previous_beta_alpha = []
    var_previous_beta_beta = []
    
    for i in range(config['k']):
        
        # load the existing parameter or create the new parameter
        if config["completed_task_idx"] is not None:
            mean_init = torch.load(
                os.path.join(completed_result_dir, 'mean{}_varapprox{}.pt'.format(config["completed_task_idx"], i)),
                map_location="cpu"
            )
            covar_init = torch.load(
                os.path.join(completed_result_dir, 'covar{}_varapprox{}.pt'.format(config["completed_task_idx"], i)),
                map_location="cpu"
            )
            approx = VariationalApprox_v2(
                config['device'], config["num_mc_sample"], mean_init=mean_init, 
                covar_init=covar_init, init_optim_lrsch=False
            )
        else:
            model_temp = getattr(models, config['net'])(**config['net_kwargs'])
            approx = VariationalApprox_v2(
                config['device'], config["num_mc_sample"], model=model_temp, init_optim_lrsch=False
            )
        
        approx.update_mean_cov()
        var_approx_list.append(approx)
        
    
    # define the variational object of beta
    alpha = config["alpha"] if config['completed_task_idx'] is None else \
        np.load(os.path.join(completed_result_dir, "alpha{}.npy".format(config['completed_task_idx'])))

    var_beta = var_approx_beta(alpha, config['k'], config['beta_num_mc_sample'], config["device"], init_optim_lrsch=False, implicit=config["implicit"])

    # sparsity_vec = evidential_sparsity(var_beta, var_previous_beta_alpha, var_previous_beta_beta, config['eta'])
    # delete_num = np.sum(np.where(sparsity_vec==1))
    # print(sparsity_vec)
    # print(delete_num)
    # print(aaa)
    # load the checkpoint if existing
    if config['completed_task_idx'] is not None:
        prev_glob_step \
            = torch.load(os.path.join(completed_result_dir, 'prev_glob_step{}.pt'.format(config['completed_task_idx'])))
        evalset = torch.load(
            os.path.join(completed_result_dir, 'evalset{}.pt'.format(config['completed_task_idx'])))
    else:
        prev_glob_step = 0
        evalset = []

    print("=============={}================".format(len(evalset)))

    # get all the dataset name
    num_dataset_to_run = len(config['dataset_ls']) if config['num_dataset_to_run'] == 'all' \
        else config['num_dataset_to_run']
    
    for task_idx, task in enumerate(config['dataset_ls'][:num_dataset_to_run], 0):
        
        if config['increase_num'] and task_idx != 0:

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

        print(config['k'])
        
        # init the berns
        # var_beta.update_bern()

        # train the model with different dataset
        if config['completed_task_idx'] is not None and config['completed_task_idx'] >= task_idx:
            # pass the completed dataset
            pass
        else:
            # get the split file of the current task
            split_dir = os.path.join(os.path.join(config['data_dir'], task), config['split_folder'])

            # define optimiser and lr scheduler for each variational objective in the list
            for var_approx in var_approx_list:
                var_approx.optimizer = getattr(optim, config[task]['optim_outer_name']) \
                    (list(var_approx.mean.values()) + list(var_approx.covar.values()), **config[task]['optim_outer_kwargs'])
                if config[task]['lr_sch_outer_name'] is None:
                    var_approx.lr_scheduler = None
                else:
                    var_approx.lr_scheduler = getattr(lr_scheduler, config[task]['lr_sch_outer_name']) \
                        (optim_outer, **config[task]['lr_sch_outer_kwargs'])


            var_beta.optimizer = getattr(optim, config[task]['optim_outer_name']) \
                    (var_beta.parameters(), **config[task]['optim_outer_kwargs'])
            if config[task]['lr_sch_outer_name'] is None:
                var_beta.lr_scheduler = None
            else:
                var_beta.lr_scheduler = getattr(lr_scheduler, config[task]['lr_sch_outer_name']) \
                    (optim_outer, **config[task]['lr_sch_outer_kwargs'])

            # define transformation of images
            transformation = transforms.Compose(
                enlist_transformation(img_resize=config['img_resize'], is_grayscale=config['is_grayscale'],
                                      device=config['device'], img_normalise=config[task]['img_normalise'])
            )

            # define the training dataset
            trainset = FewShotImageDataset(
                    task_list=np.load(os.path.join(split_dir, 'metatrain.npy'), allow_pickle=True).tolist(),
                    supercls=config[task]['supercls'], img_lvl=int(config[task]['supercls']) + 1, transform=transformation,
                    relabel=None, device=var_approx.device, cuda_img_tensor=config['cuda_img_tensor'],
                    verbose='{} trainset'.format(task)
            )

            # define & append meta-evaluation dataset and dataloader
            evalset.append(FewShotImageDataset(
                task_list=np.load(os.path.join(split_dir, 'metatest.npy'), allow_pickle=True).tolist(),
                supercls=config[task]['eval_supercls'], img_lvl=int(config[task]['eval_supercls']) + 1,
                transform=transformation, relabel=None, device=config['device'],
                cuda_img_tensor=config['cuda_img_tensor'], verbose='{} evalset'.format(task)
            ))

            # training the model
            metatrain_seqdataset_multi_amvi(
                model=model, var_approx_list=var_approx_list, var_beta=var_beta, model_device=config["device"], 
                trainset=trainset, evalset=evalset, outer_kl_scale=config[task]['outer_kl_scale'],
                nstep_outer=config[task]['nstep_outer'], nstep_inner=config[task]['nstep_inner'],
                lr_inner=config[task]['lr_inner'], first_order=config[task]['first_order'], seqtask=config['seqtask'],
                num_task_per_itr=config[task]['num_task_per_itr'], task_by_supercls=config[task]['task_by_supercls'],
                num_way=config['net_kwargs']['num_way'], num_shot=config[task]['num_shot'],
                num_query_per_cls=config[task]['num_query_per_cls'], eval_prev_task=True,
                eval_per_num_iter=config[task]['eval_per_num_iter'], num_eval_task=config[task]['num_eval_task'],
                eval_task_by_supercls=config[task]['eval_task_by_supercls'],
                nstep_inner_eval=config[task]['nstep_inner_eval'], writer=writer, task_idx=task_idx,
                prev_glob_step=prev_glob_step, verbose=task, sample_mean=config["sample_mean"], sample_batch=config["sample_batch"],
                max_eval = config['max_eval']
            )

            # update global step
            prev_glob_step += config[task]['nstep_outer'] 

            
            # save the previous beta distribution
            var_previous_beta_alpha.append(var_beta.var_gamma1.data.cpu().numpy())
            var_previous_beta_beta.append(var_beta.var_gamma2.data.cpu().numpy())
            

            if var_beta is not None:
                gamma = np.vstack([var_beta.var_gamma1.detach().cpu().numpy(), \
                    var_beta.var_gamma2.detach().cpu().numpy()])
                # print(gamma)
                np.save(os.path.join(result_dir, "alpha{}_nonsparse.npy".format(task_idx)), gamma)

                # rho = var_beta.bern_pro.detach().cpu().numpy()
                # np.save(os.path.join(result_dir, "rho{}.npy".format(task_idx)), rho)

            for i,var_approx in enumerate(var_approx_list):
                torch.save(var_approx.mean, f=os.path.join(result_dir, 'mean{}_varapprox{}__nonsparse.pt'.format(task_idx, i)))
                torch.save(var_approx.covar, f=os.path.join(result_dir, 'covar{}_varapprox{}__nonsparse.pt'.format(task_idx, i)))

            # evidential sparsity
            if config['sparsity'] and len(var_approx_list) != 1:
                sparsity_vec = evidential_sparsity(var_beta, var_previous_beta_alpha, var_previous_beta_beta, config['eta'])
                delete_num = np.sum(np.where(sparsity_vec==0))

                # update the beta
                var_beta.delete_cluster(sparsity_vec)
                config['k'] -= delete_num

                # update the theta
                delete_index = np.argmin(sparsity_vec)
                pop_index = np.where(sparsity_vec==0)[0]
                for i in range(1,len(pop_index)+1):
                    var_approx_list.pop(pop_index[-i])

            # save the model
            torch.save(prev_glob_step, f=os.path.join(result_dir, 'prev_glob_step{}.pt'.format(task_idx)))
            torch.save(evalset, f=os.path.join(result_dir, 'evalset{}.pt'.format(task_idx)))
            for i,var_approx in enumerate(var_approx_list):
                torch.save(var_approx.mean, f=os.path.join(result_dir, 'mean{}_varapprox{}.pt'.format(task_idx, i)))
                torch.save(var_approx.covar, f=os.path.join(result_dir, 'covar{}_varapprox{}.pt'.format(task_idx, i)))
            
            if var_beta is not None:
                gamma = np.vstack([var_beta.var_gamma1.detach().cpu().numpy(), \
                    var_beta.var_gamma2.detach().cpu().numpy()])
                # print(gamma)
                np.save(os.path.join(result_dir, "alpha{}.npy".format(task_idx)), gamma)

                # rho = var_beta.bern_pro.detach().cpu().numpy()
                # np.save(os.path.join(result_dir, "rho{}.npy".format(task_idx)), rho)

            # update mean and covariance of meta-parameters
            var_beta.update_posterior(len(var_approx_list))
            for var_approx in var_approx_list:
                var_approx.update_mean_cov() 

        torch.cuda.empty_cache()

    # check how long it ran
    run_time_print = '\ncompleted in {}'.format(datetime.datetime.now() - start_datetime)
    os.system('echo "{}"'.format(run_time_print)) if system() == 'Linux' else print(run_time_print)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('BOMVI Sequential Dataset')
    parser.add_argument('--config_path', type=str, help='Path of .json file to import config from')
    args = parser.parse_args()
    # load config file
    jsonfile = open(str(args.config_path))
    config = json.loads(jsonfile.read())
    # train
    train(config=config, run_spec=os.path.splitext(os.path.split(args.config_path)[-1])[0], seed=random.getrandbits(24))
