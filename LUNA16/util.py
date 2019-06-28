"""
util - other functions
"""
import torch
import numpy as np
from common import Config

I = torch.FloatTensor(np.eye(Config.batch_size),)
E = torch.FloatTensor(np.ones((Config.batch_size, Config.batch_size)))
normalize_1 = Config.batch_size
normalize_2 = Config.batch_size * Config.batch_size - Config.batch_size

def kl_loss_function(output1, output2, p):
    """
    :param output1: left output
    :param output2: right output
    :param p: priori
    :return: -MIG^f where f-divergence is KL divergence
    """
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    noise = torch.rand(1)*0.0001
    m1 = torch.log(m*I+ I*noise + E - I)
    m2 = m*(E-I)
    return -(sum(sum(m1)) + Config.batch_size)/normalize_1 + sum(sum(m2)) / normalize_2

def pearson_loss_function(output1, output2, p):
    """
    :param output1: left output
    :param output2: right output
    :param p: priori
    :return: -MIG^f where f-divergence is Pearson X^2
    """
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    m1 = m*I
    m2 = m*(E-I)
    m2 = m2*m2
    return -(2 * sum(sum(m1)) - 2 * Config.batch_size) / normalize_1 + (sum(sum(m2)) - normalize_2) / normalize_2

def js_loss_function(output1, output2, p):
    """
    :param output1: left output
    :param output2: right output
    :param p: priori
    :return: -MIG^f where f-divergence is Jensen-Shannon
    """
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    noise = torch.rand(1)*0.0001
    m1 = torch.log(2*m*I/(E+m*I)+ I*noise + E - I)
    m2 = -torch.log((E - I)*2/(E+m) +(E - I) * noise + I)
    return -(sum(sum(m1)))/normalize_1 + sum(sum(m2)) / normalize_2

def M_step(expert_label, mu):
    """
    :param expert_label: all experts' predictions with size [batch_size, expert_num]
    :param mu: updating parameters with size[batch_size, num_classes]
    :return: updated parameters for experts with size [expert_num, num_classes, num_classes]
    """
    normalize = torch.sum(mu, 0).float()
    expert_label = expert_label.long()
    expert_parameters = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
    for i in range(mu.size()[0]):
        for R in range(Config.expert_num):
            expert_parameters[R, :, expert_label[i, R]] += mu[i].float()
    expert_parameters = expert_parameters / normalize.unsqueeze(1)
    expert_parameters = expert_parameters.cuda()
    return expert_parameters
