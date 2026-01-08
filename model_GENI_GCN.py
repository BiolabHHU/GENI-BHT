import torch
import torch.nn as nn

import torch.nn.functional as F

from GENI_layer import GENI

import torch


def edge_feature(x, mask,Tp):

    epsilon = 1e-6
    ones = torch.ones_like(mask)
    mask = torch.where(mask != 0, ones, mask)# 保证 mask 中只有 0 和 1 两种值
    eps = 1e-8  # 避免除 0 / log(0)
    cov = torch.matmul(x, x.permute(0,2,1))
    energy_bn = x.pow(2).sum(dim=2, keepdim=True)  # →  (B, N, 1)
    amplitude_bn = energy_bn.sqrt()
    energy_mat = torch.matmul(amplitude_bn,amplitude_bn.permute(0,2,1))  # (B, N, N)
    FC = cov / (energy_mat + eps)  # (B, N, N)
    FC_clamped = torch.clamp(FC, -0.99999, 0.99999)

    feature = 0.5 * torch.log((1 + FC_clamped) / (1 - FC_clamped))  # (B, N, N)
    feature = feature * mask

    epsilon = torch.tensor(epsilon, dtype=feature.dtype).cuda()
    feature = torch.where((feature == 0) & (mask ==1), epsilon, feature)

    feature[mask == 0] =0
    N, _, _ = feature.shape
    nonzero_elements_list = []

    # 遍历每个样本
    for i in range(N):

        adj_matrix = feature[i]
        edge_index = torch.nonzero(torch.triu(adj_matrix,diagonal=1))
        b = edge_index.T
        e = adj_matrix[b[0, :], b[1, :]]

        nonzero_elements_list.append(e.view(-1, 1))
    edge_feature = torch.stack(nonzero_elements_list)

    return edge_feature





class Encode(nn.Module):
    def __init__(self,n_in,n_hid,n_out,Tp):
        super(Encode, self).__init__()
        # n_in:1 n_hid:2  n_out:3

        self.ces0 = GENI(n_out, n_out*2,n_in,n_in, node_layer=True) #lyl node_layer: 一个布尔值，决定当前这个实例是“节点层”还是“边层”
        self.ces1 = GENI(n_out*2, n_out*2,n_hid,n_hid*2, node_layer=False)#lyl 边层
        self.ces2 = GENI(n_out*2, n_out*2, n_hid*2, n_hid*2, node_layer=True)#lyl 节点层
        self.Tp = Tp


    def forward(self, data,edge,adj,adj_e,T):


        x_v, x_e = self.ces0(data,edge,adj,adj_e,T)


        x_e1 = edge_feature(x_v, adj, self.Tp)
        x_e = torch.cat([x_e,x_e1],dim=2)
        x_v, x_e = self.ces1(F.relu(x_v),x_e,adj,adj_e,T)
        x_v, x_e = self.ces2(F.relu(x_v),F.relu(x_e), adj, adj_e, T)
        return x_v, x_e

# Decode model
class Decode(nn.Module):
    def __init__(self,n_in,n_hid,n_out):
        super(Decode, self).__init__()

        self.ces0 = GENI(n_out*2, n_out*2, n_hid*2, n_in, node_layer=False)
        self.ces1 = GENI(n_out*2, n_out, n_in, n_in, node_layer=True)


    def forward(self, data,edge,adj,adj_e,T):

        x_v = data
        x_e = edge
        x_v,x_e = self.ces0(x_v,x_e,adj,adj_e,T)
        x_v,x_e = self.ces1(F.relu(x_v),F.relu(x_e),adj,adj_e,T)

        return x_v, x_e


class Classify(nn.Module):
    def __init__(self, num_of_hidden,num_of_hidden_classify,node_m,edge_m):
        super(Classify, self).__init__()

        #lyl num_of_hidden 40 , num_of_hidden_classify 20
        self.layer1 = nn.Linear(node_m*6, num_of_hidden_classify)
        self.layer2 = nn.Linear(edge_m*4,num_of_hidden_classify)

        self.num_of_hidden = num_of_hidden
        self.num_of_hidden_classify = num_of_hidden_classify

        # lyl num_of_hidden 40 , num_of_hidden_classify 20 ,
        self.classifier_0 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden,
                                                     out_features=self.num_of_hidden_classify, bias=True),
                                           nn.ReLU())

        # residual
        self.classifier_1 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          )
        # residual
        self.classifier_2 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          )
        # residual
        self.classifier_3 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          )

        self.classifier_out = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                  out_features=2, bias=True))


    def forward(self, node,edge):


        x_v = self.layer1(node.flatten(1))
        x_e = self.layer2(edge.flatten(1))
        x = torch.cat([x_v,x_e],dim=1)


        y0 = self.classifier_0(x)

        y1 = self.classifier_1(y0)+ y0
        y2 = self.classifier_2(y1)+ y1
        y3 = self.classifier_3(y2)+ y2

        out = self.classifier_out(y3 + y2 +y1 + y0)

        return out,x


class GENI_GCN(nn.Module):
    def __init__(self, n_in,n_hid,n_out, num_of_hidden_classify,node_m,edge_m,Tp):
        super(GENI_GCN, self).__init__()
        self.encode = Encode(n_in,n_hid,n_out,Tp)
        self.decode = Decode(n_in,n_hid,n_out)
        self.classify = Classify(40,num_of_hidden_classify,node_m,edge_m)

    #lyl n_in 1, n_hid 2 , n_out 3, num_of_hidden_classify 20
    def forward(self,data,edge,adj,adj1,T):

        x_v,x_e = self.encode(data,edge,adj,adj1,T)
        x_v1,x_e1 = self.decode(x_v,x_e,adj,adj1,T)
        c,x = self.classify(x_v,x_e)
        return x,c,x_v1,x_e1
