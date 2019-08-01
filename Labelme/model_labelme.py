# model - models for different methods
import torch
import torch.nn as nn
import torch.nn.functional as F
from common_labelme import Config
from data_labelme import expert_tmatrix,train_loader
import numpy as np
import vgg

torch.cuda.set_device(Config.device_id)


p_pure = torch.FloatTensor([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])
p = p_pure
p_mbem = p_pure


#p = torch.FloatTensor([0.5,0.5])
#base_model.add(Flatten(input_shape=data_train_vgg16.shape[1:]))
#base_model.add(Dense(128, activation='relu'))
#base_model.add(Dropout(0.5))
#base_model.add(Dense(N_CLASSES))
#base_model.add(Activation("softmax"))

class left_neural_net_labelme(nn.Module):
    def __init__(self):
        super(left_neural_net_labelme, self).__init__()
        self.linear1 = nn.Linear(8192,128)
        self.linear2 = nn.Linear(128,8)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.resize(x.size()[0],8192)
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.linear2(x)

        return torch.nn.functional.softmax(x,dim=1)


class left_neural_net_mw_labelme(nn.Module):
    def __init__(self):
        super(left_neural_net_mw_labelme, self).__init__()

        for i in range(Config.expert_num):
            m_name = "mw" + str(i+1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))
        self.weights_init()
        self.linear1 = nn.Linear(8192,128)
        self.linear2 = nn.Linear(128,8)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.resize(x.size()[0],8192)
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.linear2(x)

        x = torch.nn.functional.softmax(x, dim=1)
        out = F.log_softmax(self.mw1(x),1).unsqueeze(0)
        for name, module in self.named_children():
            if name[0:2] != 'mw' or name == 'mw1':
                continue
            out = torch.cat((out,F.log_softmax(module(x), 1).unsqueeze(0)),0)
        return out, x

    def weights_init(self):
        for name, module in self.named_children():
            if name[0] == 'm':
                module.weight.data = torch.eye(Config.num_classes)

class left_neural_net_dn_labelme(nn.Module):
    def __init__(self):
        super(left_neural_net_dn_labelme, self).__init__()
        for i in range(Config.expert_num):
            m_name = "dn" + str(i+1)
            self.add_module(m_name,nn.Linear(128, Config.num_classes))
        self.linear1 = nn.Linear(8192, 128)
        self.linear2 = nn.Linear(128, 8)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.resize(x.size()[0],8192)
        x = self.dropout1(F.relu(self.linear1(x)))

        out = F.log_softmax(self.dn1(x),1).unsqueeze(0)
        for name, module in self.named_children():
            if name[0:2] != 'dn' or name == 'dn1':
                continue
            out = torch.cat((out,F.log_softmax(module(x), 1).unsqueeze(0)),0)
        return out

class left_neural_net_mw(nn.Module):
    def __init__(self):
        super(left_neural_net_mw, self).__init__()

        for i in range(Config.expert_num):
            m_name = "mw" + str(i+1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))
        self.weights_init()
        self.vgg =vgg.VGG('VGG16')
        self.features = self.vgg.features
        self.classifier = self.vgg.classifier
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.nn.functional.softmax(x, dim=1)
        out = F.log_softmax(self.mw1(x),1).unsqueeze(0)
        for name, module in self.named_children():
            if name[0:2] != 'mw' or name == 'mw1':
                continue
            out = torch.cat((out,F.log_softmax(module(x), 1).unsqueeze(0)),0)
        return out, x

    def weights_init(self):
        for name, module in self.named_children():
            if name[0] == 'm':
                module.weight.data = torch.eye(Config.num_classes)


class left_neural_net_dn(nn.Module):
    def __init__(self):
        super(left_neural_net_dn, self).__init__()
        for i in range(Config.expert_num):
            m_name = "dn" + str(i+1)
            self.add_module(m_name,nn.Linear(512, Config.num_classes))
        self.vgg = vgg.VGG('VGG16')
        self.features = self.vgg.features
        self.classifier = self.vgg.classifier
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        out = F.log_softmax(self.dn1(x),1).unsqueeze(0)
        for name, module in self.named_children():
            if name[0:2] != 'dn' or name == 'dn1':
                continue
            out = torch.cat((out,F.log_softmax(module(x), 1).unsqueeze(0)),0)
        return out


class right_neural_net_EM(nn.Module):
    def __init__(self,prior):
        super(right_neural_net_EM, self).__init__()
        #self.priority = prior.unsqueeze(1).cuda()
        self.priority = prior.cuda()
        self.p = nn.Linear(1,2,bias=False)
        for i in range(Config.expert_num):
            m_name = "fc" + str(i+1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))

        self.weights_init()

    def forward(self, x, left_p, prior = 0, type=0) :

        entity = torch.ones((x.size()[0],1)).cuda()
        out = 0
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            #print(module(x[:, index-1, :]))
            out += module(x[:, index-1, :])
        #print("Out:  ",out)
        #priority =  self.p(entity)
        if type == 1 :
            out += torch.log(left_p+0.001) + torch.log(self.priority)
        elif type == 2 :
            out += torch.log(self.priority)
        elif type == 3 :
            out += torch.log(left_p + 0.001)
        return torch.nn.functional.softmax(out,dim=1)

    def weights_init(self):
        for name, module in self.named_children():
            if name == 'p':
                module.weight.data = self.priority
                continue
            index = int(name[2:])
            module.weight.data = torch.log(expert_tmatrix[index - 1] + 0.01)


    def weights_update(self, expert_parameters):
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            module.weight.data = torch.log(expert_parameters[index - 1] + 0.0001)


    def get_prior(self):
        for name, module in self.named_children():
            if name == 'p':
                return module.weight.data


# models and optimizers for different methods
net_mw = left_neural_net_mw_labelme().cuda()
net_dn = left_neural_net_dn_labelme().cuda()
left_model = left_neural_net_labelme().cuda()
left_model_pure = left_neural_net_labelme().cuda()

left_model_supervised = left_neural_net_labelme().cuda()
left_model_em = left_neural_net_labelme().cuda()

left_model_mbem = left_neural_net_labelme().cuda()
right_model_mbem = right_neural_net_EM(p_mbem).cuda()


right_model = right_neural_net_EM(p).cuda()
right_model_pure = right_neural_net_EM(p_pure).cuda()
right_model_em = right_neural_net_EM(p).cuda()

left_model_majority = left_neural_net_labelme().cuda()




net_mw_optimizer = torch.optim.Adam(net_mw.parameters(), lr=Config.left_learning_rate)
net_dn_optimizer = torch.optim.Adam(net_dn.parameters(), lr=Config.left_learning_rate)
left_optimizer = torch.optim.Adam(left_model.parameters(), lr=Config.left_learning_rate)
left_optimizer_pure = torch.optim.Adam(left_model_pure.parameters(), lr=Config.left_learning_rate)
left_optimizer_majority = torch.optim.Adam(left_model_majority.parameters(), lr = Config.left_learning_rate)
left_optimizer_supervised = torch.optim.Adam(left_model_supervised.parameters(), lr = Config.left_learning_rate)
left_optimizer_em = torch.optim.Adam(left_model_em.parameters(), lr = Config.left_learning_rate)
right_optimizer = torch.optim.Adam(right_model.parameters(), lr = Config.right_learning_rate)
right_optimizer_pure = torch.optim.Adam(right_model_pure.parameters(), lr = Config.right_learning_rate)
right_optimizer_EM = torch.optim.Adam(right_model.parameters(), lr = Config.right_learning_rate  * 0.1 )
left_optimizer_EM = torch.optim.Adam(left_model.parameters(), lr = Config.left_learning_rate * 0.01 )
left_optimizer_mbem = torch.optim.Adam(left_model_mbem.parameters(), lr = Config.left_learning_rate)