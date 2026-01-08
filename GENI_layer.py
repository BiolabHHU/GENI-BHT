import math
import numpy as np
import torch
import time

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def normalize_adj(edge,add_selfloops):


    if add_selfloops:
        a = torch.eye((edge.size(1))).cuda()
        edge = edge + a  # Add self-loops
    adj = torch.abs(edge)
    deg = torch.sum(adj, dim=2)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)

    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt, offset=0, dim1=-2, dim2=-1)

    edge_norm = torch.matmul(deg_inv_sqrt, edge)
    edge_norm = torch.matmul(edge_norm, deg_inv_sqrt)

    return edge_norm





class GENI(Module):


    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GENI, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        #lyl 节点层
        if node_layer:
            print("this is a node layer")
            self.node_layer = True
            self.weight = Parameter(torch.Tensor(in_features_v, out_features_v))


            self.pe = Parameter(torch.Tensor(in_features_e,1))
            self.pv = Parameter(torch.Tensor(in_features_v,1))

            self.gate_U = Parameter(torch.Tensor(2,1))
            if bias:
                self.bias = Parameter(torch.Tensor(out_features_v))

            else:
                self.register_parameter('bias', None)
        # lyl 边层
        else:
            print("this is an edge layer")
            self.node_layer = False
            self.weight = Parameter(torch.Tensor(in_features_e, out_features_e))

            self.pe = Parameter(torch.Tensor(in_features_e,1))
            self.pv = Parameter(torch.Tensor(in_features_v,1))

            self.gate_U = Parameter(torch.Tensor(2, 1))
            if bias:
                self.bias = Parameter(torch.Tensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


        stdv_pv = 1. / math.sqrt(self.pv.size(0))
        self.pv.data.uniform_(-stdv_pv, stdv_pv)
        stdv_pe = 1. / math.sqrt(self.pe.size(0))
        self.pe.data.uniform_(-stdv_pe, stdv_pe)

        self.gate_U.data.uniform_(-0.1, 0.1)


    def forward(self, H_v, H_e, adj_v, adj_e, T):

        if self.node_layer:
            # 1. A_edge = T@diag(H_e @ pe) @ T^T + diag(H_v @ pv)
            H1v = torch.matmul(H_v.float(),self.pv)
            H2v = torch.diag_embed(H1v.squeeze(-1))

            H1e = torch.matmul(H_e.float(), self.pe)
            H2e = torch.diag_embed(H1e.squeeze(-1))
            H3e = torch.matmul(T.float(),  H2e)
            H4e = torch.matmul(H3e, T.float().permute(0, 2, 1))

            H5  =  H2v + H4e

            # 1. A_edge = normalize( A_edge * (I+A) )

            I = torch.eye((adj_v.size(1))).cuda()  # lyl (30, 30)单位矩阵
            adj_v1 = adj_v + I  # 邻接矩阵对角线变1
            A_edge = H5 * adj_v1
            A_edge = normalize_adj(A_edge, False)

            # 2. A_node = normalize(A_v + I)
            A_node = normalize_adj(adj_v, True)

            # 3. Gate: G = sigmoid(U [A_edge || A_node] V)先把 (A_edge, A_node)


            A_stack = torch.stack([A_edge, A_node], dim=-1)  # (B, Nv, Nv, 2)


            G_logits = torch.matmul( A_stack,self.gate_U).squeeze(-1)  # (B, Nv, Nv)

            G = torch.sigmoid(G_logits)

            # 4. A_fused = G ⊙ A_edge + (1 - G) ⊙ A_node

            A_fused_v = G * A_edge + (1 - G) * A_node

            # 5. Propagate

            output = torch.matmul(A_fused_v, torch.matmul(H_v.float(), self.weight))


            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e  # ret = ( adjusted_A @( H_v @ Weight ) )

        else:

            # 1. A_node = T^T@diag(H_v @ pv) @ T + diag(H_e @ pe)

            H21e = torch.matmul(H_e.float(), self.pe)
            H22e = torch.diag_embed(H21e.squeeze(-1))

            H21v = torch.matmul(H_v.float(), self.pv)
            H22v = torch.diag_embed(H21v.squeeze(-1))

            H23v = torch.matmul(T.float().permute(0, 2, 1), H22v)
            H24v = torch.matmul(H23v, T.float())

            A_node = H22e + H24v

            # 1. A_node = normalize( A_node * (I+A) )

            I = torch.eye((adj_e.size(1))).cuda()
            adj_e1 = adj_e + I

            A_node = A_node * adj_e1

            A_node = normalize_adj(A_node , False)

            # 2. A_edge = normalize(A_e + I)
            A_edge = normalize_adj(adj_e ,True)

            # 3. Gate: G = sigmoid(U [A_node||A_edge  ] V)
            A_stack = torch.stack([A_node,A_edge], dim=-1)  # (B, Ne, Ne, 2)

            G_logits = torch.matmul(A_stack,self.gate_U).squeeze(-1)# (B, Ne,Ne)

            G = torch.sigmoid(G_logits)

            # 4. A_fused = G ⊙ A_edge + (1 - G) ⊙ A_node

            A_fused_e = G * A_edge + (1 - G) * A_node

            # 5. Propagate

            output = torch.matmul(A_fused_e, torch.matmul(H_e.float(), self.weight))





            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

