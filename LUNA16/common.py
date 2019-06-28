"""
common - configurations
"""
import numpy as np
from torchvision.transforms import transforms
import argparse

parser = argparse.ArgumentParser(description='MaxMIG')
parser.add_argument('--case', type=int, metavar='N', help='case')
parser.add_argument('--device', type=int, metavar='N', help='case')
parser.add_argument('--expertise', type=int, metavar='N', help='case')
parser.add_argument('--path', type=str,
                    help='path to your dataset')
args = parser.parse_args()

class Config:
    data_root = args.data
    training_size = 6484
    test_size = 1622
    lexpert = [[.6,.4],[.4,.6]]
    if args.expertise == 0:
        if args.case == 1:
            expert_num = 10
        elif args.case == 2:
            expert_num = 25
        elif args.case == 3:
            expert_num = 12
        as_expertise = np.array([lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert,lexpert])
    elif args.expertise == 1:
        if args.case == 1:
            expert_num = 5
        elif args.case == 2:
            expert_num = 10
        elif args.case == 3:
            expert_num = 10
        as_expertise = np.array(
            [[[0.6,0.4],[0.1,0.9]],
             [[0.9,0.1],[0.4,0.6]],
             [[0.6,0.4],[0.3,0.7]],
             [[0.7,0.3],[0.3,0.7]],
             [[0.7,0.3],[0.4,0.6]]])

    missing_label = np.array([0, 0, 0, 0, 0])
    missing = False
    num_classes = 2
    left_input_size = 28 * 28
    batch_size = 16
    left_learning_rate = 1e-4
    right_learning_rate = 1e-4
    epoch_num = 20
    device_id = args.device
    experiment_case = args.case

    train_transform = transforms.Compose([
            transforms.Resize((150, 150),interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
    test_transform = transforms.Compose([
            transforms.Resize((150, 150),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
