# model - models for different methods
import torch
import torch.nn as nn
import torch.nn.functional as F
from dogsvscats.common import Config
from dogsvscats.data import expert_tmatrix

torch.cuda.set_device(Config.device_id)
p = torch.FloatTensor([0.5,0.5])

class left_neural_net(nn.Module):
    """
    the common architecture for the left model
    """
    def __init__(self):
        super(left_neural_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv1_batch_norm = nn.BatchNorm2d(32)
        self.classifier  = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, Config.num_classes),
        )

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = self.conv1_batch_norm(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)

        return torch.nn.functional.softmax(x,dim=1)


class left_neural_net_log(nn.Module):
    """
    the common architecture for the left model with log_softmax
    """
    def __init__(self):
        super(left_neural_net_log, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv1_batch_norm = nn.BatchNorm2d(32)
        self.classifier  = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, Config.num_classes),
        )

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = self.conv1_batch_norm(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class left_neural_net_cl(nn.Module):
    """
    left neural net for Crowds Layer method
    """
    def __init__(self):
        super(left_neural_net_cl, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv1_batch_norm = nn.BatchNorm2d(32)
        for i in range(Config.expert_num):
            m_name = "mw" + str(i+1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))
        self.classifier  = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, Config.num_classes),
        )
        self.weights_init()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = self.conv1_batch_norm(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
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
    """
    left neural net for Docter Net method
    """
    def __init__(self):
        super(left_neural_net_dn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv1_batch_norm = nn.BatchNorm2d(32)
        for i in range(Config.expert_num):
            m_name = "dn" + str(i+1)
            self.add_module(m_name,nn.Linear(128, Config.num_classes))
        self.classifier  = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Dropout(),
        )
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = self.conv1_batch_norm(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        out = F.log_softmax(self.dn1(x),1).unsqueeze(0)
        for name, module in self.named_children():
            if name[0:2] != 'dn' or name == 'dn1':
                continue
            out = torch.cat((out,F.log_softmax(module(x), 1).unsqueeze(0)),0)
        return out


class right_neural_net(nn.Module):
    """
    right neural net for max-mig
    also as a EM updater (without SGD) for AggNet
    """
    def __init__(self,prior):
        super(right_neural_net, self).__init__()
        self.priority = prior.unsqueeze(1).cuda()
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
            out += module(x[:, index-1, :])
        priority =  self.p(entity)
        if type == 1 :
            out += torch.log(left_p+0.001) + torch.log(priority)
        elif type == 2 :
            out += torch.log(priority)
        elif type == 3 :
            out += torch.log(left_p + 0.001)
        return torch.nn.functional.softmax(out,dim=1)

    def weights_init(self):
        for name, module in self.named_children():
            if name == 'p':
                module.weight.data = self.priority
                continue
            index = int(name[2:])
            module.weight.data = torch.log(expert_tmatrix[index - 1] + 0.0001).cuda()

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
left_model_majority = left_neural_net_log().cuda()
net_cl = left_neural_net_cl().cuda()
net_dn = left_neural_net_dn().cuda()
left_model_agg = left_neural_net().cuda()
right_model_agg = right_neural_net(p).cuda()
left_model_mig = left_neural_net().cuda()
right_model_mig = right_neural_net(p).cuda()
left_model_true = left_neural_net_log().cuda()

left_optimizer_majority = torch.optim.Adam(left_model_majority.parameters(), lr = Config.left_learning_rate)
net_cl_optimizer = torch.optim.Adam(net_cl.parameters(), lr=Config.left_learning_rate)
net_dn_optimizer = torch.optim.Adam(net_dn.parameters(), lr=Config.left_learning_rate)
left_optimizer_agg = torch.optim.Adam(left_model_agg.parameters(), lr = Config.left_learning_rate)
left_optimizer_mig = torch.optim.Adam(left_model_mig.parameters(), lr=Config.left_learning_rate)
right_optimizer_mig = torch.optim.Adam(right_model_mig.parameters(), lr = Config.right_learning_rate)
left_optimizer_true = torch.optim.Adam(left_model_true.parameters(), lr = Config.left_learning_rate)
