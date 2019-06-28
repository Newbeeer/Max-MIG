"""
common - configurations
"""
import torch
import numpy as np
import argparse
from torchvision.transforms import transforms
parser = argparse.ArgumentParser(description='CoTraining')

parser.add_argument('--device', type=int, metavar='N',
                    help='case')
parser.add_argument('--c1', default="", metavar='DIR', help='path to check left')
parser.add_argument('--c2', default="", metavar='DIR', help='path to check right')
args = parser.parse_args()
class Config:
    data_root = './dogdata'
    #data_root  = '/data1/xuyilun/LUNA16/data'
    training_size = 10000
    test_size = 1188
    as_expertise = np.array([[0.6, 0.8, 0.7, 0.6, 0.7],[0.6,0.6,0.7,0.9,0.6]])


    missing_label = np.array([0, 0, 0, 0, 0])
    missing = True

    num_classes = 8
    batch_size = 64
    left_learning_rate = 1e-4
    right_learning_rate = 1e-4
    epoch_num = 20
    #########################
    expert_num = 59
    device_id = args.device
    experiment_case = 3
    log_case = 1
    #########################
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
    test_transform = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])

