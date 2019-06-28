"""
util - other functions
"""
import torch
import numpy as np
from common_cifar import Config
from torch.autograd import Variable

I = torch.FloatTensor(np.eye(Config.batch_size),)
E = torch.FloatTensor(np.ones((Config.batch_size, Config.batch_size)))
normalize_1 = Config.batch_size
normalize_2 = Config.batch_size * Config.batch_size - Config.batch_size

def mig_loss_function(output1, output2, p):
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    noise = torch.rand(1)*0.0001
    m1 = torch.log(m*I+ I*noise + E - I)
    m2 = m*(E-I)
    return -(sum(sum(m1)) + Config.batch_size)/normalize_1 + sum(sum(m2)) / normalize_2

def tvd_loss_function(output1, output2, p):
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    noise = torch.rand(1)*0.0001
    m1 = torch.log(m*I + I * noise + E - I)
    m2 = torch.log(m*(E-I) + I )

    return -(sum(sum(torch.sign(m1))))/normalize_1 + sum(sum(torch.sign(m2))) / normalize_2

def pearson_loss_function(output1, output2, p):
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))

    m1 = m*I
    m2 = m*(E-I)
    m2 = m2*m2
    return -(2 * sum(sum(m1)) - 2 * Config.batch_size) / normalize_1 + (sum(sum(m2)) - normalize_2) / normalize_2

def reverse_kl_loss_function(output1, output2, p):
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    m1 = m*I
    m1 = -I/(m1.float() + E - I)
    m2 = torch.log(m*(E-I) + I)
    return -(sum(sum(m1)))/normalize_1 + (-sum(sum(m2)) - normalize_2) / normalize_2

def sh_loss_function(output1, output2, p):
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    m1 = m*I
    m1 = torch.sqrt(I/(m1.float() + E - I))
    m2 = torch.sqrt(m*(E-I))
    return -(-sum(sum(m1)) + Config.batch_size)/normalize_1 + sum(sum(m2)) / normalize_2

def entropy_loss(outputs):
    num = outputs.size()[0]
    temp = -outputs * torch.log(outputs+0.0001)
    loss = torch.sum(temp)
    loss /= num
    return loss

def M_step(expert_label,mu):

    #---------------------------------------------------------------#
    #                                                               #
    # expert_label size : batch_size * expert_num                   #
    # mu : batch_size * num_classes                                 #
    # expert_parameters = expert_num * num_classes * num_classes    #
    #                                                               #
    #---------------------------------------------------------------#

    if not Config.missing:
        normalize = torch.sum(mu, 0).float()
        expert_label = expert_label.long()
        expert_parameters = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
        for i in range(mu.size()[0]):
            for R in range(Config.expert_num):
                expert_parameters[R, :, expert_label[i, R]] += mu[i].float()

        expert_parameters = expert_parameters / normalize.unsqueeze(1)
    else:
        normalize = torch.zeros(Config.expert_num,Config.num_classes)
        expert_label = expert_label.long()
        expert_parameters = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
        for i in range(mu.size()[0]):
            for R in range(Config.expert_num):
                if expert_label[i,R] < 0:
                    continue
                expert_parameters[R, :, expert_label[i, R]] += mu[i].float()
                normalize[R] += mu[i].float()

        normalize = normalize + 1 * (normalize == 0).float()

        for R in range(Config.expert_num):
            expert_parameters[R] = expert_parameters[R] / normalize[R].unsqueeze(1)

    expert_parameters = expert_parameters.cuda()
    return expert_parameters

def M_step_p_mbem(t):

    p = torch.zeros(Config.num_classes)
    t = t.long()
    for i in range(t.size(0)):
        p[t[i]] += 1
    p /= t.size()[0]
    return p


def M_step_mbem(expert_label,t):

    #---------------------------------------------------------------#
    #                                                               #
    # expert_label size : batch_size * expert_num                   #
    # t : batch_size                                                #
    # expert_parameters = expert_num * num_classes * num_classes    #
    #                                                               #
    #---------------------------------------------------------------#

    normalize = torch.zeros(Config.expert_num, Config.num_classes)
    expert_label = expert_label.long()
    t = t.long()
    expert_parameters = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))


    for i in range(t.size()[0]):
        for R in range(Config.expert_num):
            if expert_label[i, R] < 0:
                continue
            expert_parameters[R, t[i], expert_label[i, R]] += 1
            normalize[R,t[i]] += 1
    normalize = normalize + 1 * (normalize == 0).float()

    for R in range(Config.expert_num):
        expert_parameters[R] = expert_parameters[R] / normalize[R].unsqueeze(1)

    expert_parameters = expert_parameters.cuda()
    return expert_parameters


def print_recons_result(right_model, confusion_matrix):

    confusion_loss = 0
    for i in range(1,len(list(right_model.parameters()))):
        para = list(right_model.parameters())[i].detach().cpu()
        #print("Expert %d" %i)
        local_confusion_matrix = torch.nn.functional.softmax(para, dim=1)
        #print(local_confusion_matrix)
        residual_matrix = local_confusion_matrix - confusion_matrix[i-1, :, :]
        residual = torch.sum(abs(residual_matrix))
        confusion_loss += residual

    print("Total variation:", confusion_loss.item())

def initial_priori(train_loader):
    p = torch.zeros((Config.num_classes))


    total = 0
    for batch_idx, (left_data, right_data, label) in enumerate(train_loader):
        linear_sum = torch.sum(right_data, dim=1)
        _, majority = torch.max(linear_sum, 1)
        majority = Variable(majority).long()
        total += label.size()[0]
        for i in range(Config.num_classes):
            p[i] += torch.sum(majority == i).float()
    p = p/float(total)
    return p

def update_priori(model, train_loader):
    # waiting for solution
    p = torch.zeros((Config.num_classes))

    # updating priori by posteri

    total = 0
    for batch_idx, (left_data, right_data, label) in enumerate(train_loader):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        outputs = model(images)
        _, predicts = torch.max(outputs.data, 1)
        total += ep.size()[0]
        predicts = predicts.detach().cpu()
        for i in range(Config.num_classes):
            p[i] += torch.sum(predicts == i).float()

    p = p/float(total)
    '''
    # updating priori by loss
    pri = priori
    pri = Variable(pri, requires_grad=True)
    loss = mig_loss_function(left_outputs.detach(),right_outputs.detach(),p)
    loss.backward()
    grad = pri.grad
    pri = pri.detach() - Config.alpha * grad
    pri = torch.exp(pri)
    pri = pri / torch.sum(pri)
    
    '''

    '''
    # true priori
    p[0] = 0.5
    p[1] = 0.5
    '''
    return p