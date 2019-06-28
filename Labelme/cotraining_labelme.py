"""
cotraining - main file for training and testing
"""

import torch.nn.functional as F
from model_cifar import *
from data_cifar import *
from util_cifar import *
from torch.autograd import Variable
from sklearn.metrics import  roc_auc_score
from tensorboardX import SummaryWriter


torch.cuda.set_device(Config.device_id)


best_cotraining = 0
best_mw = 0
best_dn = 0
best_majority = 0
best_supervised = 0
best_agg = 0
best_mbem = 0


data_loader = torch.utils.data.DataLoader(dataset=train_dataset_mbem, batch_size=Config.batch_size, shuffle=False)
def train(epoch, priori_pure, priori) :
    print('current train epoch = %d' % epoch)

    net_mw.train()
    net_dn.train()
    left_model_pure.train()
    right_model_pure.train()
    left_model_majority.train()

    majority_loss = 0
    pure_loss = 0
    co_loss1 = 0
    co_loss2 = 0
    mw_loss = 0
    dn_loss = 0

    p_pure = priori_pure
    p = priori
    print("Priori from Pure Co-training:", p_pure)

    for batch_idx , (left_data, right_data) in enumerate(train_loader):
        if left_data.size()[0] != Config.batch_size :
            continue

        ep = Variable(right_data).float().cuda()

        images = Variable(left_data).float().cuda()





        # Pure Co-Training
        right_optimizer_pure.zero_grad()
        left_optimizer_pure.zero_grad()
        left_outputs = left_model_pure(images).cpu().float()
        right_outputs = right_model_pure(ep, left_outputs, prior=p_pure, type=2).cpu().float()
        #print("Left:",left_outputs,"Right:",right_outputs)
        loss = mig_loss_function(left_outputs, right_outputs, p_pure)
        #print("Loss:",loss.item())
        loss.backward()
        right_optimizer_pure.step()
        left_optimizer_pure.step()
        pure_loss += loss.item()

        # Majority Vote
        linear_sum = torch.sum(right_data, dim=1)
        _, majority = torch.max(linear_sum, 1)
        majority = Variable(majority).long().cuda()

        left_optimizer_majority.zero_grad()
        left_outputs_majority = left_model_majority(images).float()
        left_outputs_majority = torch.log(left_outputs_majority).float()
        left_loss_majority = nn.functional.nll_loss(left_outputs_majority, majority)
        left_loss_majority.backward()
        left_optimizer_majority.step()

        # DEEP MW
        net_mw_optimizer.zero_grad()
        out, hold = net_mw(images)
        loss = 0
        label = torch.max(ep, dim=2)[1].long().cuda()


        cnt = 0
        for i in range(Config.expert_num):

            mask = (torch.max(ep[:,i,:],dim=1)[0])
            zeromap = torch.zeros((1,images.size()[0])).cuda()
            vec = F.nll_loss(out[i], label[:,i],reduce=False)
            vec = torch.where(mask == 0,zeromap,vec)
            loss += torch.sum(vec)
            cnt += torch.sum(mask > 0)


        loss = (1.0 / cnt.item()) * loss
        loss.backward()
        net_mw_optimizer.step()

        # DEEP DN
        net_dn_optimizer.zero_grad()
        out = net_dn(images)
        loss = 0
        cnt = 0

        for i in range(Config.expert_num):

            mask = (torch.max(ep[:,i,:],dim=1)[0])
            zeromap = torch.zeros((1,images.size()[0])).cuda()
            vec = F.nll_loss(out[i], label[:,i],reduce=False)
            vec = torch.where(mask == 0,zeromap,vec)
            loss += torch.sum(vec)
            cnt += torch.sum(mask > 0)

        loss = (1.0 / (cnt.item())) * loss
        loss.backward()
        net_dn_optimizer.step()


    #p_pure = update_priori(left_model_pure, train_loader)
    #p = update_priori(left_model, train_loader)
    p_pure = torch.squeeze(right_model_pure.get_prior().detach().cpu())


    return p_pure, p

loss_f = torch.nn.KLDivLoss(size_average=False)

def train_agg():
    train_loader_em = torch.utils.data.DataLoader(dataset=train_dataset_em, batch_size=Config.batch_size, shuffle=True)

    # training
    left_model_em.train()
    right_model_em.train()
    em_loss = 0
    for batch_idx, (left_data, right_data,label) in enumerate(train_loader_em):
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()

        left_optimizer_em.zero_grad()
        left_outputs = left_model_em(images)
        left_outputs = torch.log(left_outputs)
        loss = loss_f(left_outputs,label)/Config.batch_size
        loss.backward()
        left_optimizer_em.step()
        em_loss += loss
    print('Agg loss: {:.4f}'.format(em_loss))

    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_em(images)
        right_outputs = right_model_em(ep, left_outputs, type=3)
        
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_em.label_update(right_label)

    # M-step
    right_outputs_all = []
    ep_label_all = []
    for batch_idx, (left_data, right_data,label) in enumerate(data_loader):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_em(images)
        right_outputs = right_model_em(ep, left_outputs, type=3)
        right_outputs = list(right_outputs.detach().cpu().numpy())
        right_outputs_all += right_outputs

        expert_label = torch.zeros((ep.size()[0], ep.size()[1])).cuda()
        max_element,expert_label[:, :] = torch.max(ep,2)
        mask = (max_element == 0)
        expert_label = expert_label + (-Config.num_classes) * mask.float().cuda()
        expert_label = list(expert_label.detach().cpu().numpy())
        ep_label_all += expert_label

    right_outputs_all = torch.FloatTensor(right_outputs_all)
    #instance_num * num_classes
    ep_label_all = torch.FloatTensor(ep_label_all)
    #instance_num * expert_num
    expert_parameters = M_step(ep_label_all, right_outputs_all)
    #exoert_num * num_classes * num_classes
    right_model_em.weights_update(expert_parameters)


def train_mbem():
    train_loader_mbem = torch.utils.data.DataLoader(dataset=train_dataset_mbem, batch_size=Config.batch_size, shuffle=True)

    # training
    left_model_mbem.train()
    right_model_mbem.train()

    for batch_idx, (left_data, right_data,label) in enumerate(train_loader_mbem):
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()
        left_optimizer_mbem.zero_grad()
        left_outputs = left_model_mbem(images)
        left_outputs = torch.log(left_outputs)
        loss = loss_f(left_outputs,label)/Config.batch_size
        loss.backward()
        left_optimizer_mbem.step()


    # estimate confusion matrices and prior class distribution q given t
    right_all_data = []
    left_label_all = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader):

        right_all_data += list(right_data.detach().cpu().numpy())
        images = Variable(left_data).float().cuda()
        left_outputs = torch.max(left_model_mbem(images),1)[1]
        left_label_all += list(left_outputs.detach().cpu().numpy())


    right_all_data = torch.FloatTensor(right_all_data)
    mask,right_all_data = torch.max(right_all_data,2)
    mask = (mask == 0)
    right_all_data = right_all_data + (-Config.num_classes) * mask.long()
    # instance_num * expert_num

    left_label_all = torch.FloatTensor(left_label_all)
    # instance_num


    expert_parameters = M_step_mbem(right_all_data, left_label_all)
    # exoert_num * num_classes * num_classes
    right_model_mbem.weights_update(expert_parameters)
    p_mbem = M_step_p_mbem(left_label_all).cuda()


    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader):
        ep = Variable(right_data).float().cuda()
        right_outputs = right_model_mbem(ep, p_mbem, type=3)
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_mbem.label_update(right_label)




def test(epoch) :
    print('current test epoch = %d' % epoch)


    net_mw.eval()
    net_dn.eval()
    left_model_supervised.eval()
    left_model_majority.eval()
    left_model_em.eval()
    left_model_pure.eval()
    right_model_pure.eval()
    left_model_mbem.eval()

    total_corrects = 0
    total_sample = 0
    auc_label = []
    auc_output = []

    total_corrects_pure = 0
    auc_pure_output = []

    total_corrects_mw = 0
    auc_mw_output = []

    total_corrects_dn = 0
    auc_dn_output = []

    total_corrects_majority = 0
    auc_majority_output = []

    total_corrects_em = 0
    auc_em_output = []

    total_corrects_mbem = 0

    for images, ep in test_loader:
        images = Variable(images).float().cuda()
        labels = ep.cuda()
        auc_label += list(labels)
        total_sample += images.size()[0]


        # Pure Cotraining
        outputs = left_model_pure(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_pure += torch.sum(predicts == labels)

        #MBEM
        outputs = left_model_mbem(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_mbem += torch.sum(predicts == labels)

        # Majority Vote
        outputs = left_model_majority(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_majority += torch.sum(predicts == labels)

        # Agg
        outputs = left_model_em(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_em += torch.sum(predicts == labels)

        # DEEP MW
        out, outputs = net_mw(images)
        auc_mw_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_mw += torch.sum(predicts == labels)

        # DEEP DN
        out = net_dn(images)
        outputs = 0
        for i in range(Config.expert_num):
            outputs += torch.exp(out[i])
        outputs = outputs/ Config.expert_num
        auc_dn_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_dn += torch.sum(predicts == labels)

    cotrain_acc = float(total_corrects_pure)/float(total_sample)
    majority_acc = float(total_corrects_majority) / float(total_sample)
    dn_acc = float(total_corrects_dn) / float(total_sample)
    mw_acc = float(total_corrects_mw) / float(total_sample)
    agg_acc = float(total_corrects_em) / float(total_sample)
    mbem_acc = float(total_corrects_mbem) / float(total_sample)


    #print("Pure Cotraining ACC:",float(total_corrects_pure)/float(total_sample))
    #print("Majority Voting ACC:", float(total_corrects_majority) / float(total_sample))
    #print("DN Cotraining ACC:", float(total_corrects_dn) / float(total_sample))
    #print("MW Cotraining ACC:", float(total_corrects_mw) / float(total_sample))

    return cotrain_acc,majority_acc,dn_acc,mw_acc,agg_acc,mbem_acc


if __name__ == '__main__':


    #p_pure = initial_priori(train_loader)
    #p = initial_priori(train_loader)
    for epoch in range(Config.epoch_num):

        p_pure, p = train(epoch, p_pure, p)
        #train_agg()
        #train_mbem()
        print("--------")
        co,mv,dn,mw, agg,mbem= test(epoch)
        best_cotraining = max(best_cotraining,co)
        best_dn = max(best_dn,dn)
        best_majority = max(best_majority,mv)
        best_mw = max(best_mw,mw)
        best_agg = max(best_agg,agg)

        print("Pure Cotraining ACC:", best_cotraining)
        print("Majority Voting ACC:", best_majority)
        print("DN Cotraining ACC:", best_dn)
        print("MW Cotraining ACC:", best_mw)
        print("AGG Cotraining ACC:", best_agg)
        print("MBEM Cotraining ACC:", mbem)