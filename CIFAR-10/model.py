# model - models for different methods
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import Config
from data import expert_tmatrix

torch.cuda.set_device(Config.device_id)
priori_fixed = torch.FloatTensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

class VGG(nn.Module):
    """
    the common architecture for the left model
    """
    def __init__(self, vgg_name):
        super(VGG, self).__init__()

        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.features = self._make_layers(self.cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out,dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class left_neural_net_cl(nn.Module):
    """
    left neural net for Crowds Layer method
    """
    def __init__(self):
        super(left_neural_net_cl, self).__init__()

        for i in range(Config.expert_num):
            m_name = "mw" + str(i+1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))
        self.weights_init()
        self.vgg =VGG('VGG16')
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
                index = int(name[2:])
                module.weight.data = torch.log(expert_tmatrix[index - 1] + 0.0001)

class left_neural_net_dn(nn.Module):
    """
    left neural net for Docter Net method
    """
    def __init__(self):
        super(left_neural_net_dn, self).__init__()
        for i in range(Config.expert_num):
            m_name = "dn" + str(i+1)
            self.add_module(m_name,nn.Linear(512, Config.num_classes))
        self.vgg = VGG('VGG16')
        self.features = self.vgg.features
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
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
    def __init__(self, prior):
        super(right_neural_net, self).__init__()
        self.priority = prior.cuda()
        for i in range(Config.expert_num):
            m_name = "fc" + str(i+1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))
        self.weights_init()

    def forward(self, x, left_p, prior = 0, type=0):
        out = 0
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            out += module(x[:, index-1, :])
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
            module.weight.data = torch.log(expert_tmatrix[index - 1] + 0.0001)

    def weights_update(self, expert_parameters):
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            module.weight.data = torch.log(expert_parameters[index - 1] + 0.0001)

# models and optimizers for different methods
left_model_majority = VGG('VGG16').cuda()
net_cl = left_neural_net_cl().cuda()
net_dn = left_neural_net_dn().cuda()
left_model_agg = VGG('VGG16').cuda()
right_model_agg = right_neural_net(priori_fixed).cuda()
left_model_mig = VGG('VGG16').cuda()
right_model_mig = right_neural_net(priori_fixed).cuda()
left_model_true = VGG('VGG16').cuda()

left_optimizer_majority = torch.optim.Adam(left_model_majority.parameters(), lr = Config.left_learning_rate)
net_cl_optimizer = torch.optim.Adam(net_cl.parameters(), lr=Config.left_learning_rate)
net_dn_optimizer = torch.optim.Adam(net_dn.parameters(), lr=Config.left_learning_rate)
left_optimizer_agg = torch.optim.Adam(left_model_agg.parameters(), lr = Config.left_learning_rate)
left_optimizer_mig = torch.optim.Adam(left_model_mig.parameters(), lr=Config.left_learning_rate)
right_optimizer_mig = torch.optim.Adam(right_model_mig.parameters(), lr = Config.right_learning_rate)
left_optimizer_true = torch.optim.Adam(left_model_true.parameters(), lr = Config.left_learning_rate)
