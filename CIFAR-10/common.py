"""
common - configurations
"""
import numpy as np
from torchvision.transforms import transforms
import argparse

parser = argparse.ArgumentParser(description='MaxMIG')
parser.add_argument('--case', type=int, metavar='N',
                    help='experiment case')
parser.add_argument('--device', type=int, metavar='N',
                    help='device number')
parser.add_argument('--expertise', type=int, metavar='N',
                    help='expertise, 0 for low, 1 for high')
parser.add_argument('--path', type=str,
                    help='path to your dataset')
args = parser.parse_args()

class Config:
    data_root = args.path
    device_id = args.device
    experiment_type = args.expertise
    experiment_case = args.case

    epoch_num = 50
    training_size = 50000
    test_size = 10000
    num_classes = 10
    batch_size = 64
    left_learning_rate = 1e-3
    right_learning_rate = 1e-4
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if experiment_type == 0:
        if experiment_case == 1:
            expert_num = 10

        elif experiment_case == 2:
            senior = 10
            expert_num = 15

        elif experiment_case == 3:
            senior = 10
            junior_1 = 1
            junior_2 = 1
            expert_num = 12

        as_expertise = np.zeros((expert_num, num_classes, num_classes))
        for i in range(10):
            for j in range(10):
                as_expertise[i][j][j] = 0.2

        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if k == j:
                        continue
                    as_expertise[i][j][k] = (1 - as_expertise[i][j][j]) / 9

    else:
        if experiment_case == 1:
            expert_num = 5

        elif experiment_case == 2:
            senior = 5
            expert_num = 10

        elif experiment_case == 3:
            senior = 5
            junior_1 = 2
            junior_2 = 3
            expert_num = 10

        as_expertise = np.zeros((expert_num, num_classes, num_classes))
        if experiment_type == 'L':
            if experiment_case == 1:
                senior = 10
                expert_num = 10

            elif experiment_case == 2:
                senior = 10
                expert_num = 15

            elif experiment_case == 3:
                senior = 10
                junior_1 = 1
                junior_2 = 1
                expert_num = 12

        # five senior experts:
        # cat&dog deer&horse airplane&bird automobile&trunk frog&ship are pairs that are difficult for experts to distinguish,
        # but anyone can easily distinguish between pairs. e.g.between cat&deer
        # five senior experts give their different solutions

        # expert 0: she always labels a fixed one in the pair
        # e.g. she always labels "cat" when she sees cat/dog
        as_expertise[0][0][0] = 1
        as_expertise[0][2][0] = 1
        as_expertise[0][1][1] = 1
        as_expertise[0][9][1] = 1
        as_expertise[0][3][3] = 1
        as_expertise[0][5][3] = 1
        as_expertise[0][4][4] = 1
        as_expertise[0][7][4] = 1
        as_expertise[0][6][6] = 1
        as_expertise[0][8][6] = 1

        # expert 1: she labels one of the two classes in a pair randomly
        # e.g. she labels "cat" with prob=0.5, "dog" with prob=0.5 when she sees cat/dog
        as_expertise[1][0][0] = 0.5
        as_expertise[1][0][2] = 0.5
        as_expertise[1][2][0] = 0.5
        as_expertise[1][2][2] = 0.5
        as_expertise[1][1][1] = 0.5
        as_expertise[1][1][9] = 0.5
        as_expertise[1][9][1] = 0.5
        as_expertise[1][9][9] = 0.5
        as_expertise[1][3][3] = 0.5
        as_expertise[1][3][5] = 0.5
        as_expertise[1][5][3] = 0.5
        as_expertise[1][5][5] = 0.5
        as_expertise[1][4][4] = 0.5
        as_expertise[1][4][7] = 0.5
        as_expertise[1][7][4] = 0.5
        as_expertise[1][7][7] = 0.5
        as_expertise[1][6][6] = 0.5
        as_expertise[1][6][8] = 0.5
        as_expertise[1][8][6] = 0.5
        as_expertise[1][8][8] = 0.5

        # expert 2: she is familiar with mammals, so she can distinguish cat&dog deer&horse
        # but for other pairs, she acts like expert 1
        as_expertise[2][0][0] = 0.5
        as_expertise[2][0][2] = 0.5
        as_expertise[2][2][0] = 0.5
        as_expertise[2][2][2] = 0.5
        as_expertise[2][1][1] = 0.5
        as_expertise[2][1][9] = 0.5
        as_expertise[2][9][1] = 0.5
        as_expertise[2][9][9] = 0.5
        as_expertise[2][3][3] = 1
        as_expertise[2][5][5] = 1
        as_expertise[2][4][4] = 1
        as_expertise[2][7][7] = 1
        as_expertise[2][6][6] = 0.5
        as_expertise[2][6][8] = 0.5
        as_expertise[2][8][6] = 0.5
        as_expertise[2][8][8] = 0.5

        # expert 3: she is familiar with vehicles, so she can distinguish airplane&bird automobile&trunk frog&ship
        # but for other pairs, she acts like expert 0
        as_expertise[3][0][0] = 1
        as_expertise[3][2][2] = 1
        as_expertise[3][1][1] = 1
        as_expertise[3][9][9] = 1
        as_expertise[3][6][6] = 1
        as_expertise[3][8][8] = 1
        as_expertise[3][3][3] = 1
        as_expertise[3][5][3] = 1
        as_expertise[3][4][4] = 1
        as_expertise[3][7][4] = 1

        # expert 4: she gives the right prediction with prob=0.6 and make error in pairs with prob=0.4
        as_expertise[4][0][0] = 0.6
        as_expertise[4][0][2] = 0.4
        as_expertise[4][2][0] = 0.4
        as_expertise[4][2][2] = 0.6
        as_expertise[4][1][1] = 0.6
        as_expertise[4][1][9] = 0.4
        as_expertise[4][9][1] = 0.4
        as_expertise[4][9][9] = 0.6
        as_expertise[4][3][3] = 0.6
        as_expertise[4][3][5] = 0.4
        as_expertise[4][5][3] = 0.4
        as_expertise[4][5][5] = 0.6
        as_expertise[4][4][4] = 0.6
        as_expertise[4][4][7] = 0.4
        as_expertise[4][7][4] = 0.4
        as_expertise[4][7][7] = 0.6
        as_expertise[4][6][6] = 0.6
        as_expertise[4][6][8] = 0.4
        as_expertise[4][8][6] = 0.4
        as_expertise[4][8][8] = 0.6
