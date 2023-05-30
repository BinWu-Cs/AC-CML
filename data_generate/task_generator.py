import os
import glob
import random
import numpy as np
import warnings
from tqdm import tqdm

class TaskGenerator(object):
    def __init__(self, num_way, supercls=True, split_dir='../data/omniglot/split', tasklist_dir='./experiment/noname'):
        self.num_way = num_way
        self.supercls = supercls
        self.split_dir = split_dir
        self.tasklist_dir = tasklist_dir

        if not os.path.exists(self.tasklist_dir):
            os.makedirs(self.tasklist_dir)

    def generate_task_list(self, split_npyfilename='metatrain.npy', num_class_excl=None, num_task_per_supercls=None, save_npy=True):
        '''
        Generate task lists
        :param classdirs_list: directory where the train, val, test classes are
        :param num_class_excl: num of classes to change after every episode;
            only applicable for online training. Set to None if offline.
        :param task_by_tier: only applicable if num_class_excl==None; True if each task must be classes from the same tier.
        :param num_episode: number of tasks ie. number of episodes;
            only applicable for offline training or val & test
        :param num_shuffle: number of shuffles before sampling tasks
        :param save_npy: True to save task list as .npy
        :return: list of tasks directories if save_npy is False
        '''
        # check if tasklist already exist
        if os.path.exists(os.path.join(self.tasklist_dir, 'tasklist.npy')):
            warnings.warn('"tasklist.npy" file already exists. Existing tasklist will be reused.')
            pass
        else:
            classdir_list = np.load(os.path.join(self.split_dir, split_npyfilename), allow_pickle=True).tolist()
            task_dir_list = []
            if self.supercls:
                # shuffle tier list
                random.shuffle(classdir_list)
                # generate task for each tier
                for sampled_tier in tqdm(classdir_list, desc='Generating task list'):
                    tier_classdir_list = glob.glob(sampled_tier + '/*')
                    # shuffle class dirs
                    random.shuffle(tier_classdir_list)
                    # generate breakpoint indices for the classes
                    breakpoint_indices = [self.num_way]
                    for _ in range((len(tier_classdir_list) - self.num_way) // num_class_excl):
                        breakpoint_indices.append(breakpoint_indices[-1] + num_class_excl)
                    # list for task updates
                    tier_classdir_list_split = np.array_split(tier_classdir_list, breakpoint_indices)
                    # concatenate sequentially into tasks
                    sampled_task = tier_classdir_list_split[0].tolist()
                    task_dir_list.append(sampled_task)
                    if num_task_per_supercls is not None:
                        if num_task_per_supercls > 1:
                            for cls_to_include in tier_classdir_list_split[1:num_task_per_supercls]:
                                if len(cls_to_include) > 0:
                                    classes_to_include = cls_to_include.tolist()
                                    classes_to_remain = sampled_task[len(classes_to_include):]
                                    sampled_task = classes_to_remain + classes_to_include

                                    task_dir_list.append(sampled_task)
                                else:
                                    pass
                        else:
                            pass
                    else:
                        for cls_to_include in tier_classdir_list_split[1:]:
                            if len(cls_to_include) > 0:
                                classes_to_include = cls_to_include.tolist()
                                classes_to_remain = sampled_task[len(classes_to_include):]
                                sampled_task = classes_to_remain + classes_to_include

                                task_dir_list.append(sampled_task)
                            else:
                                pass
            else:
                random.shuffle(classdir_list)
                # generate breakpoint indices for the classes
                breakpoint_indices = [self.num_way]
                for _ in range((len(classdir_list) - self.num_way) // num_class_excl):
                    breakpoint_indices.append(breakpoint_indices[-1] + num_class_excl)
                # list for task updates
                classdir_list_split = np.array_split(classdir_list, breakpoint_indices)
                # concatenate sequentially into tasks
                sampled_task = classdir_list_split[0].tolist()
                task_dir_list.append(sampled_task)
                for cls_to_include in tqdm(classdir_list_split[1:], desc='Generating task list'):
                    if len(cls_to_include) > 0:
                        classes_to_include = cls_to_include.tolist()
                        classes_to_remain = sampled_task[len(classes_to_include):]
                        sampled_task = classes_to_remain + classes_to_include

                        task_dir_list.append(sampled_task)
                    else:
                        pass
            if save_npy:
                np.save(os.path.join(self.tasklist_dir, 'tasklist.npy'), task_dir_list)
            else:
                return task_dir_list #, task_label_list

    def load_task(self, task_npyfilename='tasklist.npy'):
        self.tasklist = np.load(os.path.join(self.tasklist_dir, task_npyfilename), allow_pickle=True).tolist()
