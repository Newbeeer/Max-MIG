"""
main - main file for training and testing
"""
from model import *
from data import *
from util import *
from torch.autograd import Variable
from tqdm import tqdm

best_majority_acc = 0
best_cl_acc = 0
best_dn_acc = 0
best_agg_acc = 0
best_mig_acc = 0
best_true_acc = 0
best_forecast_acc = 0

def train():
    left_model_majority.train()
    net_cl.train()
    net_dn.train()
    left_model_mig.train()
    right_model_mig.train()
    left_model_true.train()

    majority_loss = 0
    cl_loss = 0
    dn_loss = 0
    mig_loss = 0
    true_loss = 0

    for batch_idx , (left_data, right_data, true_label) in enumerate(tqdm(train_loader)):
        if left_data.size()[0] != Config.batch_size :
            continue

        ep = Variable(right_data).float().cuda()
        true_label = Variable(true_label).long().cuda()
        images = Variable(left_data).float().cuda()

        # Majority Vote
        linear_sum = torch.sum(right_data,dim=1)
        _, majority = torch.max(linear_sum,1)
        majority = Variable(majority).long().cuda()
        left_optimizer_majority.zero_grad()
        outputs = left_model_majority(images).float()
        outputs = torch.log(outputs).float()
        loss = nn.functional.nll_loss(outputs, majority)
        loss.backward()
        left_optimizer_majority.step()
        majority_loss += loss

        # Crowds Layer
        net_cl_optimizer.zero_grad()
        out, hold = net_cl(images)
        loss = 0
        for i in range(Config.expert_num):
            _, label = torch.max(ep[:, i, :], dim=1)
            label = label.long().cuda()
            loss += F.nll_loss(out[i], label)
        loss = (1.0 / Config.expert_num) * loss
        loss.backward()
        net_cl_optimizer.step()
        cl_loss += loss

        # Docter Net
        net_dn_optimizer.zero_grad()
        out = net_dn(images)
        loss = 0
        for i in range(Config.expert_num):
            _, label = torch.max(ep[:, i, :], dim=1)
            label = label.long().cuda()
            loss += F.nll_loss(out[i], label)
        loss = (1.0 / Config.expert_num) * loss
        loss.backward()
        net_dn_optimizer.step()
        dn_loss += loss

        # MAX-MIG
        right_optimizer_mig.zero_grad()
        left_optimizer_mig.zero_grad()
        left_outputs = left_model_mig(images).cpu().float()
        right_outputs = right_model_mig(ep, left_outputs, prior=priori_fixed, type=2).cpu().float()
        loss = kl_loss_function(left_outputs, right_outputs, priori_fixed)
        loss.backward()
        right_optimizer_mig.step()
        left_optimizer_mig.step()
        mig_loss += loss

        # True label
        left_optimizer_true.zero_grad()
        left_outputs = left_model_true(images)
        left_outputs = torch.log(left_outputs)
        loss = F.nll_loss(left_outputs, true_label)
        loss.backward()
        left_optimizer_true.step()
        true_loss += loss

    return majority_loss, cl_loss, dn_loss, mig_loss, true_loss

loss_f = torch.nn.KLDivLoss(size_average=False)

def train_agg():
    # AggNet

    train_loader_agg = torch.utils.data.DataLoader(dataset=train_dataset_agg, batch_size=Config.batch_size, shuffle=True)

    # training
    left_model_agg.train()
    right_model_agg.train()
    agg_loss = 0

    for batch_idx, (left_data, right_data, label) in enumerate(tqdm(train_loader_agg)):
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()
        left_optimizer_agg.zero_grad()
        left_outputs = left_model_agg(images)
        left_outputs = torch.log(left_outputs)
        loss = loss_f(left_outputs,label)/Config.batch_size
        loss.backward()
        left_optimizer_agg.step()
        agg_loss += loss

    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(tqdm(data_loader_agg)):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_agg(images)
        right_outputs = right_model_agg(ep, left_outputs, type=3)
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_agg.label_update(right_label)

    # M-step
    right_outputs_all = []
    ep_label_all = []
    for batch_idx, (left_data, right_data, label) in enumerate(tqdm(data_loader_agg)):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_agg(images)
        right_outputs = right_model_agg(ep, left_outputs, type=3)
        right_outputs = list(right_outputs.detach().cpu().numpy())
        right_outputs_all += right_outputs

        expert_label = torch.zeros((ep.size()[0], ep.size()[1])).cuda()
        max_element,expert_label[:, :] = torch.max(ep,2)
        mask = (max_element == 0)
        expert_label = expert_label + (-Config.num_classes) * mask.float().cuda()
        expert_label = list(expert_label.detach().cpu().numpy())
        ep_label_all += expert_label

    right_outputs_all = torch.FloatTensor(right_outputs_all)
    ep_label_all = torch.FloatTensor(ep_label_all)
    expert_parameters = M_step(ep_label_all, right_outputs_all)
    right_model_agg.weights_update(expert_parameters)

    return agg_loss

def test():
    left_model_majority.eval()
    net_cl.eval()
    net_dn.eval()
    left_model_agg.eval()
    left_model_mig.eval()
    right_model_mig.eval()
    left_model_true.eval()

    total_sample = 0
    total_corrects_majority = 0
    total_corrects_cl = 0
    total_corrects_dn = 0
    total_corrects_agg = 0
    total_corrects_mig = 0
    total_corrects_true = 0
    total_corrects_forecast = 0

    for batch_idx, (images, ep, labels) in enumerate(tqdm(test_loader)):
        images = Variable(images).float().cuda()
        ep = Variable(ep).float().cuda()
        labels = labels.long()
        total_sample += images.size()[0]
        labels = labels.cuda()

        # Majority Vote
        outputs = left_model_majority(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_majority += torch.sum(predicts == labels)

        # Crowds Layer
        out, outputs = net_cl(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_cl += torch.sum(predicts == labels)

        # Doctor Net
        out = net_dn(images)
        outputs = 0
        for i in range(Config.expert_num):
            outputs += torch.exp(out[i])
        outputs = outputs / Config.expert_num
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_dn += torch.sum(predicts == labels)

        # AggNet
        outputs = left_model_agg(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_agg += torch.sum(predicts == labels)

        # MAX-MIG - aggregated forecaster
        left_outputs = left_model_mig(images)
        right_outputs = right_model_mig(ep, left_outputs, prior=priori_fixed, type=2)
        pr = priori_fixed.cuda()
        outputs = (left_outputs * right_outputs) / pr
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_forecast += torch.sum(predicts == labels)

        # MAX-MIG - soft classifier
        outputs = left_model_mig(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_mig += torch.sum(predicts == labels)

        # True label
        outputs = left_model_true(images)
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_true += torch.sum(predicts == labels)

    acc_majority = float(total_corrects_majority) / float(total_sample)
    acc_cl = float(total_corrects_cl) / float(total_sample)
    acc_dn = float(total_corrects_dn) / float(total_sample)
    acc_agg = float(total_corrects_agg) / float(total_sample)
    acc_mig = float(total_corrects_mig) / float(total_sample)
    acc_true = float(total_corrects_true) / float(total_sample)
    acc_forecast = float(total_corrects_forecast) / float(total_sample)

    print('Majority acc = {:.4f}'.format(acc_majority))
    print('CrowdsLayer acc = {:.4f}'.format(acc_cl))
    print('DoctorNet acc = {:.4f}'.format(acc_dn))
    print('AggNet acc = {:.4f}'.format(acc_agg))
    print('Max-MIG soft classifier acc = {:.4f}'.format(acc_mig))
    print('True Label acc = {:.4f}'.format(acc_true))
    print('Max-MIG aggregated forecaster acc = {:.4f}'.format(acc_forecast))

    return acc_majority, acc_cl, acc_dn, acc_agg, acc_mig, acc_true, acc_forecast

if __name__ == '__main__':
    print("True Confusion Matrix:")
    print(confusion_matrix)

    for epoch in range(Config.epoch_num):
        print("--------")
        print("case =", Config.experiment_case)
        print('current train epoch = %d' % epoch)
        majority_loss, cl_loss, dn_loss, mig_loss, true_loss = train()
        agg_loss = train_agg()
        print('Majority loss = {:.4f}'.format(majority_loss))
        print('CrowdsLayer loss = {:.4f}'.format(cl_loss))
        print('DoctorNet loss = {:.4f}'.format(dn_loss))
        print('AggNet loss = {:.4f}'.format(agg_loss))
        print('Max-MIG loss = {:.4f}'.format(mig_loss))
        print('True Label loss = {:.4f}'.format(true_loss))
        print("--------")
        print("case =", Config.experiment_case)
        print('current test epoch = %d' % epoch)
        acc_majority, acc_cl, acc_dn, acc_agg, acc_mig, acc_true, acc_forecast = test()
        best_majority_acc = max(best_majority_acc, acc_majority)
        best_cl_acc = max(best_cl_acc, acc_cl)
        best_dn_acc = max(best_dn_acc, acc_dn)
        best_agg_acc = max(best_agg_acc, acc_agg)
        best_mig_acc = max(best_mig_acc, acc_mig)
        best_true_acc = max(best_true_acc, acc_true)
        best_forecast_acc = max(best_forecast_acc, acc_forecast)
        print("--------")
        print("**best accuracy so far**")
        print('Majority acc = {:.4f}'.format(best_majority_acc))
        print('CrowdsLayer acc = {:.4f}'.format(best_cl_acc))
        print('DoctorNet acc = {:.4f}'.format(best_dn_acc))
        print('AggNet acc = {:.4f}'.format(best_agg_acc))
        print('Max-MIG soft classifier acc = {:.4f}'.format(best_mig_acc))
        print('True Label acc = {:.4f}'.format(best_true_acc))
        print('Max-MIG aggregated forecaster acc = {:.4f}'.format(best_forecast_acc))
