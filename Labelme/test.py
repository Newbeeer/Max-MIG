import numpy as np
from common_cifar import Config

N_CLASSES = Config.num_classes
def load_data(filename):
    with open(filename, 'rb') as f:
        data = np.load(f)

    f.close()
    return data

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def get_data(train):
    if train:
        path = './prepared/'

        data_train_vgg16 = load_data(path + "data_train_vgg16.npy").transpose(0, 3, 1, 2)
        print(data_train_vgg16.shape)

        #print("\nLoading AMT data...")
        answers = load_data(path + "answers.npy")
        label_train = load_data(path + "labels_train.npy")
        print(answers.shape)
        print(label_train.shape)
        N_ANNOT = answers.shape[1]

        #print("N_ANNOT:", N_ANNOT)

        answers_bin_missings = []

        for i in range(len(answers)):
            row = []
            for r in range(N_ANNOT):
                if answers[i, r] == -1:
                    row.append(0 * np.ones(N_CLASSES))
                else:
                    # print(answers[i,r])
                    row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
            answers_bin_missings.append(row)

        answers_bin_missings = np.array(answers_bin_missings)
        #print(answers_bin_missings.shape)

        # print(answers_bin_missings[0])
        return data_train_vgg16, answers_bin_missings,label_train
    else:
        path = './prepared/'

        data_test_vgg16 = load_data(path + "data_test_vgg16.npy").transpose(0, 3, 1, 2)

        labels_test = load_data(path + "labels_test.npy")

        return data_test_vgg16, labels_test




#print(label[0])