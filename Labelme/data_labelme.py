"""
data - to generate data from crowds
"""
import numpy as np
import torch
import torch.utils
from test import get_data
from tqdm import tqdm
from PIL import Image
import os
from common_labelme import Config
from torchvision import datasets


class Im_EP_labelme(torch.utils.data.Dataset):
    """
    Im_EP - to generate a dataset with images, experts' predictions and labels for learning from crowds settings
    """
    def __init__(self, as_expertise, root_path, missing_label, train):
        self.as_expertise = as_expertise
        self.class_num = Config.num_classes
        self.expert_num = Config.expert_num
        self.root_path = root_path
        self.train = train
        self.missing_label = missing_label
        if self.train:

            self.left_data, self.right_data, self.label = self.generate_data()
        else:

            self.left_data, self.right_data = self.generate_data()

    def __getitem__(self, index):
        if self.train:
            left, right, label = self.left_data[index], self.right_data[index],self.label[index]
            return left, right,label
        else:
            left, right, label = self.left_data[index], self.right_data[index],self.label[index]
            return left, right

    def __len__(self):
        if self.train:
            return Config.training_size
        else:
            return Config.test_size

    def generate_data(self):

        if self.train:
            left_data,right_data,label = get_data(True)
            return left_data, right_data, label
        else:
            left_data, right_data = get_data(False)
            return left_data,right_data



    def label_initial(self):
        linear_sum = torch.sum(torch.tensor(self.right_data), dim=1)
        linear_sum /= torch.sum(linear_sum,1).unsqueeze(1)
        self.label = linear_sum

    def label_update(self, new_label):
        self.label = new_label


class Im_EP_labelme_em(torch.utils.data.Dataset):
    """
    Im_EP - to generate a dataset with images, experts' predictions and labels for learning from crowds settings
    """
    def __init__(self, as_expertise, root_path, missing_label, train):
        self.as_expertise = as_expertise
        self.class_num = Config.num_classes
        self.expert_num = Config.expert_num
        self.root_path = root_path
        self.train = train
        self.missing_label = missing_label
        self.label = 0

        if self.train:

            self.left_data, self.right_data = self.generate_data()
        else:

            self.left_data, self.right_data = self.generate_data()
        self.label_initial()
    def __getitem__(self, index):
        if self.train:
            left, right, label = self.left_data[index], self.right_data[index], self.label[index]
        else:
            left, right, label = self.left_data[index], self.right_data[index], self.label[index]
        return left, right, label

    def __len__(self):
        if self.train:
            return Config.training_size
        else:
            return Config.test_size

    def generate_data(self):

        if self.train:
            left_data,right_data = get_data(True)
        else:
            left_data, right_data = get_data(False)

        return left_data, right_data

    def label_initial(self):

        linear_sum = torch.sum(torch.tensor(self.right_data), dim=1)
        linear_sum /= torch.sum(linear_sum,1).unsqueeze(1)
        self.label = linear_sum


    def label_update(self, new_label):
        self.label = new_label

def Initial_mats():

    if not Config.missing :
        # Never reach here
        sum_majority_prob = torch.zeros((Config.num_classes))
        confusion_matrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
        expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))

        for i, (img, ep, label) in enumerate(tqdm(train_loader)):
            linear_sum = torch.sum(ep, dim=1)

            prob = linear_sum / Config.expert_num
            sum_majority_prob += torch.sum(prob, dim=0).float()

            for j in range(ep.size()[0]):
                _, expert_class = torch.max(ep[j], 1)
                linear_sum_2 = torch.sum(ep[j], dim=0)
                prob_2 = linear_sum_2 / Config.expert_num
                for R in range(Config.expert_num):
                    expert_tmatrix[R, :, expert_class[R]] += prob_2.float()
                    confusion_matrix[R, label[j], expert_class[R]] += 1

        for R in range(Config.expert_num):
            linear_sum = torch.sum(confusion_matrix[R, :, :], dim=1)
            confusion_matrix[R, :, :] /= linear_sum.unsqueeze(1)

        expert_tmatrix = expert_tmatrix / sum_majority_prob.unsqueeze(1)
        return confusion_matrix, expert_tmatrix
    else:
        sum_majority_prob = torch.zeros((Config.expert_num,Config.num_classes))

        expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
        
        for i, (img, ep) in enumerate(tqdm(train_loader)):

            for j in range(ep.size()[0]):
                linear_sum_2 = torch.sum(ep[j], dim=0)
                prob_2 = linear_sum_2 / torch.sum(linear_sum_2)

                # prob_2 : all experts' majority voting

                for R in range(Config.expert_num):
                    # If missing ....
                    if max(ep[j,R]) == 0:

                        continue
                    _,expert_class = torch.max(ep[j,R],0)
                    expert_tmatrix[R, :, expert_class] += prob_2.float()
                    sum_majority_prob[R] += prob_2.float()

        sum_majority_prob = sum_majority_prob + 1 * (sum_majority_prob == 0).float()
        for R in range(Config.expert_num):

            expert_tmatrix[R] = expert_tmatrix[R] / sum_majority_prob[R].unsqueeze(1)

        return expert_tmatrix



# datasets for training and testing
train_dataset_em = Im_EP_labelme_em(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
train_dataset_mbem = Im_EP_labelme_em(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
test_dataset_em = Im_EP_labelme_em(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=False)

train_dataset = Im_EP_labelme(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Config.batch_size, shuffle = True)

test_dataset = Im_EP_labelme(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Config.batch_size, shuffle = False)

expert_tmatrix = Initial_mats()

print("Initial expert matrix:",expert_tmatrix)


