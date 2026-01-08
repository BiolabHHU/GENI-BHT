import argparse
import random

import numpy as np
import matlab.engine
import torch
from sklearn.metrics import roc_auc_score
from model_GENI_GCN import *
import torch.optim as optim

import torch as th
import warnings

from preprocessed_data import prepare_data

warnings.filterwarnings("ignore")
eng = matlab.engine.start_matlab()


#lyl 用 argparse 模块编写的 命令行参数 解析器。
parser = argparse.ArgumentParser(description="GENI_GCN")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--n_in", type=int, default=1, help="features number")
parser.add_argument("--n_hid", type=int, default=2, help="features number")
parser.add_argument("--n_out", type=int, default=3, help="features number")
parser.add_argument("--node_num", type=int, default=50, help="nodes number")
parser.add_argument("--num_of_hidden", type=int, default=40, help="features number")
parser.add_argument("--num_of_hidden_classify", type=int, default=20, help="features number")


opt = parser.parse_args()

ctr_mse = nn.MSELoss(reduction='mean')
ctr_mse.cuda()
ctr_entropy = nn.CrossEntropyLoss()
ctr_entropy.cuda()

#device_ids = [0]
device = th.device('cuda' if th.cuda.is_available() else 'cpu')


def set_seed(seed):
    """
    设置Python、NumPy、以及PyTorch的随机种子，保证可重复性。
    """
    random.seed(seed)  # Python 内置 random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (所有可用GPU)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train_h0(train_h0_data, edge_feature_h0, adj_h0, edge_adj_h0, T_h0, train_h0_label,print_information=False):



    train_data = torch.reshape(train_h0_data,(-1,3,num_node)).permute(0,2,1).cuda().float()


    edge_feature = torch.unsqueeze(edge_feature_h0,dim=2).cuda().float()


    adj = adj_h0.cuda().float()
    edge_adj = edge_adj_h0.cuda().float()
    T = T_h0.cuda().float()
    train_label = torch.Tensor(train_h0_label).long().reshape((len(train_h0_label),)).cuda()

    model.train()

    for epoch in range(EPOCH):

        if epoch <= opt.milestone:
            current_lr = opt.lr
        elif 30 < epoch <= 60:
            current_lr = opt.lr
        elif 60 < epoch <= 90:
            current_lr = opt.lr
        elif 90 < epoch <= 120:
            current_lr = opt.lr
        else:
            current_lr = opt.lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()

        loss_sum = 0

        x,c,x_v,x_e = model(train_data, edge_feature, adj, edge_adj, T)

        loss1 = ctr_mse(x_v, train_data)
        loss11 = ctr_mse(x_e, edge_feature)
        loss2 = ctr_entropy(c, train_label)
        loss = loss1 + loss11 + loss2

        loss_sum += loss.item()

        loss.backward()


        optimizer.step()


        correct = 0
        predicted = torch.max(c.data, 1)[1]
        correct += (predicted == train_label).sum()
        if print_information:
            # template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, acc: {}'
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, acc: {}'
            print(template2.format(epoch, loss1, loss2, torch.true_divide(correct, Batch_size)))


    model.eval()

    with torch.no_grad():

        y, s, _, _= model(train_data, edge_feature, adj, edge_adj, T)


    return y


def train_h1(train_h1_data, edge_feature_h1, adj_h1, edge_adj_h1, T_h1, train_h1_label,print_information=False):


    train_data = torch.reshape(train_h1_data,(-1,3,num_node)).permute(0,2,1).cuda().float()
    edge_feature = torch.unsqueeze(edge_feature_h1, dim=2).cuda().float()
    adj = adj_h1.cuda().float()
    edge_adj = edge_adj_h1.cuda().float()
    T = T_h1.cuda().float()
    train_label = torch.Tensor(train_h1_label).long().reshape((len(train_h1_label),)).cuda()

    model.train()

    for epoch in range(EPOCH):

        if epoch <= opt.milestone:
            current_lr = opt.lr
        elif 30 < epoch <= 60:
            current_lr = opt.lr
        elif 60 < epoch <= 90:
            current_lr = opt.lr
        elif 90 < epoch <= 120:
            current_lr = opt.lr
        else:
            current_lr = opt.lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
            # print('learning rate %f' % current_lr)
            # train
        loss_sum = 0

        optimizer.zero_grad()

        x, c, x_v, x_e = model(train_data, edge_feature, adj, edge_adj, T)

        loss1 = ctr_mse(x_v, train_data)
        loss11 = ctr_mse(x_e, edge_feature)
        loss2 = ctr_entropy(c, train_label)
        loss = loss1 + loss11 + loss2

        loss_sum += loss.item()

        loss.backward()
        optimizer.step()

        correct = 0
        predicted = torch.max(c.data, 1)[1]
        correct += (predicted == train_label).sum()
        if print_information:
            # template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, acc: {}'
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, acc: {}'
            print(template2.format(epoch, loss1, loss2, torch.true_divide(correct, Batch_size)))


    model.eval()

    with torch.no_grad():
        y, s, _, _= model(train_data, edge_feature, adj, edge_adj, T)


    return y


def judge2(y_h0_x, y_h0_label, y_h1_x, y_h1_label, num_h0, num_h1):

    # h0
    yh0_np = np.array(y_h0_x.cpu())  # deeper feature in h0
    yh0_AD = np.split(yh0_np, (num_h0,))
    yh0_AD = np.array(yh0_AD, dtype=object)
    yh0_HC = np.copy(yh0_AD)
    yh0_AD = np.delete(yh0_AD, 1, axis=0)[0]
    yh0_HC = np.delete(yh0_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh0_AD_avg = np.mean(yh0_AD, axis=(0,))
    yh0_HC_avg = np.mean(yh0_HC, axis=(0,))
    yh0_all_avg = np.mean(yh0_np, axis=(0,))

    yh0_intra_AD = np.sum(np.power(np.linalg.norm((yh0_AD - yh0_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_HC = np.sum(np.power(np.linalg.norm((yh0_HC - yh0_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_all = yh0_intra_AD + yh0_intra_HC

    yh0_inter_AD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_AD_avg), axis=0, keepdims=True), 2))
    yh0_inter_HC = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_HC_avg), axis=0, keepdims=True), 2))
    yh0_inter_all = num_h0 * yh0_inter_AD + (yh0_np.shape[0] - num_h0) * yh0_inter_HC

    yh0_out_class = yh0_intra_all / yh0_inter_all

    # h1
    yh1_np = np.array(y_h1_x.cpu())  # deeper feature in h1
    yh1_AD = np.split(yh1_np, (num_h1,))
    yh1_AD = np.array(yh1_AD, dtype=object)
    yh1_HC = np.copy(yh1_AD)
    yh1_AD = np.delete(yh1_AD, 1, axis=0)[0]
    yh1_HC = np.delete(yh1_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh1_AD_avg = np.mean(yh1_AD, axis=(0,))  # h1 ADHD均值
    yh1_HC_avg = np.mean(yh1_HC, axis=(0,))  # h1 HC均值
    yh1_all_avg = np.mean(yh1_np, axis=(0,))  # 总均值

    yh1_intra_AD = np.sum(np.power(np.linalg.norm((yh1_AD - yh1_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_HC = np.sum(np.power(np.linalg.norm((yh1_HC - yh1_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_all = yh1_intra_AD + yh1_intra_HC

    yh1_inter_AD = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_AD_avg), axis=0, keepdims=True), 2))
    yh1_inter_HC = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_HC_avg), axis=0, keepdims=True), 2))
    yh1_inter_all = num_h1 * yh1_inter_AD + (yh1_np.shape[0] - num_h1) * yh1_inter_HC

    yh1_out_class = yh1_intra_all / yh1_inter_all

    # ADHD decision function
    if yh1_out_class >= yh0_out_class:
        return True
    else:
        return False


def train():
    j = 0
    k = 0
    HC2HC = 0  # input HC, judgement result HC:  HC2HC
    HC2AD = 0  # input HC, judgement result AD:  HC2AD
    AD2AD = 0
    AD2HC = 0
    pred_tyb = []  # predicted label by authors' method


    pred_real = []  # ground truth label

    for i in range(dict_data[name_of_data]):

        train_h0_data, edge_feature_h0, adj_h0, edge_adj_h0, T_h0, train_h0_label, \
            train_h1_data, edge_feature_h1, adj_h1, edge_adj_h1, T_h1, train_h1_label, num_h0, num_h1,test_h0_label = prepare_data(index=i+1, data_name = name_of_data,numF = num_node,factor=factor,numE = edge_m)

        pred_real.append(np.rint(test_h0_label))


        y_h0 = train_h0(train_h0_data, edge_feature_h0, adj_h0, edge_adj_h0, T_h0, train_h0_label,print_information=False)
        y_h1 = train_h1(train_h1_data, edge_feature_h1, adj_h1, edge_adj_h1, T_h1, train_h1_label,print_information=False)


        judge_result2 = judge2(y_h0, train_h0_label, y_h1, train_h1_label, num_h0, num_h1)

        if judge_result2:
            k += 1
        if judge_result2 == True and test_h0_label == 2:
            j += 1
            HC2HC += 1
            pred_tyb.append(2)
        if judge_result2 == True and test_h0_label == 1:
            j += 1
            AD2AD += 1
            pred_tyb.append(1)
        if judge_result2 == False and test_h0_label == 2:
            HC2AD += 1
            pred_tyb.append(1)
        if judge_result2 == False and test_h0_label == 1:
            AD2HC += 1
            pred_tyb.append(2)



        print('\n current loop:' + str(i + 1) + ' / ' + str(dict_data[name_of_data]) + '-------------')
        print('-------------' + str(j_out + 1) + ' / ' + '50' + '-------------\n')
        print('current accuracy: ' + str(k) + '/' + str(i + 1))

    tyb1 = 'AD2AD: {}, AD2HC: {}, HC2HC: {}, HC2AD: {}'
    print(tyb1.format(AD2AD, AD2HC, HC2HC, HC2AD))
    tyb2 = '1 Accuracy: {}%'
    print(tyb2.format(100 * k / dict_data[name_of_data]))
    tyb3 = '2 Sensitivity: {}%'
    sensitivity = AD2AD / (AD2AD + AD2HC)
    print(tyb3.format(100 * AD2AD / (AD2AD + AD2HC)))
    tyb4 = '3 Specificity: {}%'
    print(tyb4.format(100 * HC2HC / (HC2AD + HC2HC)))
    tyb5 = '4 Precision: {}%'
    precision = AD2AD / (AD2AD + HC2AD)
    print(tyb5.format(100 * AD2AD / (AD2AD + HC2AD)))
    tyb6 = '5 F1 score: {}%'
    print(tyb6.format(100 * 2 * sensitivity * precision / (sensitivity + precision)))
    AUC = roc_auc_score(pred_real, pred_tyb)
    print('6 AUC:{}'.format(AUC))
    print(name_of_data)

    results_txt = str(AD2AD) + '\t' + str(AD2HC) + '\t' + str(HC2HC) + '\t' + str(HC2AD) + '\t' + str(
        100 * k / dict_data[name_of_data]) + '\t' + str(100 * AD2AD / (AD2AD + AD2HC)) + '\t' + str(
        100 * HC2HC / (HC2AD + HC2HC)) + '\t' + str(100 * AD2AD / (AD2AD + HC2AD)) + '\t' + str(
        100 * 2 * sensitivity * precision / (sensitivity + precision)) + '\t' + str(AUC) + '\n'

    with open('./results/' + name_of_data + 'GENI_BHT.txt', "a+") as f:
        f.write(results_txt)


if __name__ == '__main__':



    name_list = ['Peking_data_m', 'NYU_data_m', 'KKI_data_m', 'NI_data_m', 'Peking_1_data_m']
    dict_data = {'Peking_data_m': 194, 'NYU_data_m': 216, 'KKI_data_m': 83, 'NI_data_m': 48, 'Peking_1_data_m': 85}
    EPOCH_list = {'Peking_data_m': 100, 'NYU_data_m': 100, 'KKI_data_m': 50, 'NI_data_m': 50,
                  'Peking_1_data_m': 50}  # NI:35
    Tp_list = {'Peking_data_m': 0.65, 'NYU_data_m': 0.55, 'KKI_data_m': 0.55, 'NI_data_m': 0.45,
               'Peking_1_data_m': 0.65}
    for i_out in range(2, 3):
        set_seed(17)

        name_of_data = name_list[i_out]

        Batch_size = dict_data[name_of_data] - 1
        EPOCH = EPOCH_list[name_of_data]
        Tp = Tp_list[name_of_data]
        num_node = 35
        factor = 0.3
        edge_m = 183

        for j_out in range(25):


            model = GENI_GCN(opt.n_in, opt.n_hid, opt.n_out, opt.num_of_hidden_classify,num_node,edge_m,Tp).to(device)


            model = nn.DataParallel(model)
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=opt.lr)


            train()
