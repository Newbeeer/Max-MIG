"""
data - to generate data from crowds
"""
import numpy as np
import torch
import torch.utils
from common import Config
from torchvision import datasets


class Im_EP(torch.utils.data.Dataset):
    """
    Im_EP - to generate a dataset with images, experts' predictions and true labels for learning from crowds settings
    """
    def __init__(self, as_expertise, root_path, train):
        self.as_expertise = as_expertise
        self.root_path = root_path
        self.train = train
        if self.train:
            train_dataset = datasets.CIFAR10(root=Config.data_root,train=True,transform=Config.train_transform,download=False)
            self.left_data, self.right_data, self.label = self.generate_data(train_dataset)
        else:
            test_dataset = datasets.CIFAR10(root=Config.data_root,train=False,transform=Config.test_transform,download=False)
            self.left_data, self.right_data, self.label = self.generate_data(test_dataset)

    def __getitem__(self, index):
        if self.train:
            left, right, label = self.left_data[index], self.right_data[index], self.label[index]
        else:
            left, right, label = self.left_data[index], self.right_data[index],self.label[index]
        return left, right, label

    def __len__(self):
        if self.train:
            return Config.training_size
        else:
            return Config.test_size

    def generate_data(self, dataset):
        if self.train:
            np.random.seed(1234)
        else:
            np.random.seed(4321)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        ep = np.zeros((self.__len__(), Config.expert_num), dtype=np.int)
        labels = np.zeros((self.__len__()), dtype=np.int16)
        left_data = np.zeros((self.__len__(), 3, 32, 32))
        right_data = np.zeros((self.__len__(), Config.expert_num, Config.num_classes), dtype=np.float)
        for i, data in enumerate(data_loader):
            left_data[i] = data[0]
            labels[i] = data[1]

            #Case 1: M_s senior experts
            if Config.experiment_case == 1:
                for expert in range(Config.expert_num):
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1

            #Case 2: M_s senior experts, M_j junior experts always label 0
            if Config.experiment_case == 2:
                for expert in range(Config.senior):
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1
                for expert in range(Config.senior, Config.expert_num):
                    right_data[i][expert][0] = 1

            #Case 3: M_s senior experts, M_j junior experts copies one of the experts
            if Config.experiment_case == 3:
                for expert in range(Config.senior):
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1
                for expert in range(Config.senior, Config.senior + Config.junior_1):
                    ep[i][expert] = ep[i][0]
                    right_data[i][expert][ep[i][expert]] = 1

                for expert in range(Config.senior + Config.junior_1, Config.expert_num):
                    ep[i][expert] = ep[i][2]
                    right_data[i][expert][ep[i][expert]] = 1

        return left_data, right_data, labels

    def label_initial(self):
        linear_sum = torch.sum(torch.tensor(self.right_data), dim=1)
        linear_sum /= torch.sum(linear_sum,1).unsqueeze(1)
        self.label = linear_sum

    def label_update(self, new_label):
        self.label = new_label


def Initial_mats():
    sum_majority_prob = torch.zeros((Config.num_classes))
    confusion_matrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
    expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))

    for i, (img, ep, label) in enumerate(train_loader):
        linear_sum = torch.sum(ep, dim=1)
        label = label.long()
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


# datasets for training and testing
train_dataset = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, train=True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Config.batch_size, shuffle = True)
test_dataset = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, train=False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Config.batch_size, shuffle = False)
train_dataset_agg = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, train=True)
train_dataset_agg.label_initial()
data_loader_agg = torch.utils.data.DataLoader(dataset=train_dataset_agg, batch_size=Config.batch_size, shuffle=False)
confusion_matrix, expert_tmatrix = Initial_mats()
