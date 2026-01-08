import torch

import numpy as np

import scipy.sparse as sp

import matlab.engine
eng = matlab.engine.start_matlab()




def create_transition_matrix_new(vertex_adj):
    '''create N_v * N_e transition matrix'''
    np.fill_diagonal(vertex_adj, 0)
    edge_index = np.nonzero(np.triu(vertex_adj))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    row_index = [i for sub in edge_name for i in sub]
    col_index = np.repeat([i for i in range(num_edge)], 2)

    data = np.ones(num_edge * 2)
    T = sp.csr_matrix((data, (row_index, col_index)),
               shape=(vertex_adj.shape[0], num_edge))

    return T.toarray()


def create_transition_matrix_batch(vertex_adj_batch):
    '''create N_v * N_e transition matrix for each item in the batch'''
    batch_size, num_vertices, _ = vertex_adj_batch.shape
    transition_matrices = []

    for i in range(batch_size):
        vertex_adj = vertex_adj_batch[i]
        T = create_transition_matrix_new(vertex_adj)
        transition_matrices.append(T)
    transition_matrices = np.stack(transition_matrices)

    return transition_matrices

def create_edge_adj_batch(vertex_adj_batch):
    '''
    create an edge adjacency matrix from vertex adjacency matrix for each item in the batch
    '''
    batch_size, num_vertices, _ = vertex_adj_batch.shape
    edge_adj_batch = []
    edge_name_batch = []

    for i in range(batch_size):
        vertex_adj = vertex_adj_batch[i]
        np.fill_diagonal(vertex_adj, 0)  # Set diagonal elements to 0
        edge_index = np.nonzero(np.triu(vertex_adj))
        num_edge = int(len(edge_index[0]))
        edge_name = [tuple(x) for x in zip(edge_index[0], edge_index[1])]  # Use tuples for hashability

        edge_adj = np.zeros((num_edge, num_edge))
        for u in range(num_edge):
            for v in range(u, num_edge):
                if not set(edge_name[u]).intersection(edge_name[v]):
                    edge_adj[u, v] = 0
                else:
                    edge_adj[u, v] = 1
                    # Since we are iterating over the upper triangle, we fill the lower triangle symmetrically
                edge_adj[v, u] = edge_adj[u, v]

                # Set diagonal elements to 1 (for undirected graph)
        # np.fill_diagonal(edge_adj, 1)
        np.fill_diagonal(edge_adj, 0)

        # edge_adj_batch.append(csr_matrix(edge_adj))
        edge_adj_batch.append(edge_adj)
        edge_name_batch.append(edge_name)
    edge_adj = np.stack(edge_adj_batch)
    edge_name = np.stack(edge_name_batch)

        # Return a suzhu of matrices and a list of edge names
    return edge_adj, edge_name
def build_connected_graph_from_fc(fc_matrix, max_edges):

    N = fc_matrix.shape[0]

    np.fill_diagonal(fc_matrix, 0)



    edges = [(i, j, fc_matrix[i, j]) for i in range(N) for j in range(i + 1, N)]

    edges.sort(key=lambda x: abs(x[2]), reverse=True)


    parent = list(range(N))
    rank = [0]*N

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            return True
        return False


    adj = np.zeros_like(fc_matrix)

    edge_used = 0

    for (i, j, val) in edges:
        if union(i, j):

            adj[i,j] = val
            adj[j,i] = val
            edge_used += 1

        if edge_used == N - 1:
            break


    if max_edges is not None:

        extra_capacity = max_edges - (N-1)
        count_extra = 0
        for (i, j, val) in edges:

            if adj[i,j] != 0:
                continue
            if count_extra < extra_capacity:
                # 额外加边
                adj[i,j] = val
                adj[j,i] = val
                count_extra += 1
            else:
                break

    return adj


def build_group_template_mst_knn(all_fc, factor,edge_m):

    fisher_z=True
    B, N, _ = all_fc.shape


    if fisher_z:

        fc_z = np.arctanh(np.clip(all_fc, -0.999999, 0.999999))
        group_fc = np.tanh(fc_z.mean(axis=0))
    else:
        group_fc = all_fc.mean(axis=0)


    k_total = edge_m
    k_total = max(k_total, N - 1)

    template_adj = build_connected_graph_from_fc(group_fc.copy(), max_edges=k_total)

    mask = template_adj != 0
    adj_all = np.zeros_like(all_fc)   # (B, N, N)

    for b in range(B):
        subj_fc = all_fc[b]
        adj = np.zeros_like(subj_fc)
        adj[mask] = subj_fc[mask]
        adj_all[b] = adj

    return adj_all




def edge_feature(edge, indices):
    batch_nonzero_elements = []


    for i in range(edge.shape[0]):
        a = edge[i]
        b = indices[i].T
        e = a[b[0, :], b[1, :]]
        batch_nonzero_elements.append(e)

    batch_nonzero_elements = np.stack(batch_nonzero_elements)

    return batch_nonzero_elements

def prepare_data(index, data_name, numF, factor, numE):

    train_h0_data, train_h0_edge_attr, train_h0_label,test_h0_edge_attr, train_h1_data, train_h1_edge_attr, train_h1_label,  test_h1_edge_attr, test_h0_label, test_h1_label = eng.svm_two_class(index,data_name,numF, nargout=10)

    # train data
    train_h0_data = np.array(train_h0_data)
    train_h0_edge_attr = np.array(train_h0_edge_attr)
    # test data

    test_h0_edge_attr = np.array(test_h0_edge_attr)
    # concat
    train_h0_edge_attr = np.concatenate((train_h0_edge_attr, test_h0_edge_attr), axis = 0)

    # zero = torch.zeros_like(train_h0_edge_attr)
    train_h0_edge_attr= build_group_template_mst_knn(train_h0_edge_attr,factor,numE)   # 10%
    ones = np.ones_like(train_h0_edge_attr)
    adj_h0 = np.where(train_h0_edge_attr != 0, ones, train_h0_edge_attr)  # 0，1矩阵

    edge_adj_h0, edge_name_h0 = create_edge_adj_batch(adj_h0)
    # create transition matrix
    T_h0 = create_transition_matrix_batch(adj_h0)
    # nonzero_elements按行取出
    edge_feature_h0 = edge_feature(train_h0_edge_attr, edge_name_h0)


    train_h0_data = torch.Tensor(np.array(train_h0_data))
    edge_feature_h0 = np.delete(edge_feature_h0,-1,0)
    edge_feature_h0 = torch.Tensor(edge_feature_h0)
    T_h0 = np.delete(T_h0,-1,0)
    T_h0 = torch.Tensor(T_h0)
    adj_h0 = torch.tensor(np.delete(adj_h0,-1,0))
    edge_adj_h0 = torch.tensor(np.delete(edge_adj_h0, -1, 0))

    train_h0_label = np.array(train_h0_label)


##############################################################################################

    train_h1_data = np.array(train_h1_data)
    train_h1_edge_attr = np.array(train_h1_edge_attr)


    test_h1_edge_attr = np.array(test_h1_edge_attr)
    # concat
    train_h1_edge_attr = np.concatenate((train_h1_edge_attr, test_h1_edge_attr), axis=0)
    # zero = torch.zeros_like(train_h0_edge_attr)
    train_h1_edge_attr = build_group_template_mst_knn(train_h1_edge_attr,factor,numE)  # 10%
    ones = np.ones_like(train_h1_edge_attr)
    adj_h1 = np.where(train_h1_edge_attr != 0, ones, train_h1_edge_attr)  # 0，1矩阵

    edge_adj_h1, edge_name_h1 = create_edge_adj_batch(adj_h1)
    # create transition matrix
    T_h1 = create_transition_matrix_batch(adj_h1)
    # nonzero_elements按行取出
    edge_feature_h1 = edge_feature(train_h1_edge_attr, edge_name_h1)

    ###############################################################################


    train_h1_data = torch.Tensor(np.array(train_h1_data))
    edge_feature_h1 = np.delete(edge_feature_h1, -1, 0)
    edge_feature_h1 = torch.Tensor(edge_feature_h1)
    T_h1 = np.delete(T_h1, -1, 0)
    T_h1 = torch.Tensor(T_h1)
    adj_h1 = torch.tensor(np.delete(adj_h1, -1, 0))
    edge_adj_h1 = torch.tensor(np.delete(edge_adj_h1, -1, 0))

    train_h1_label = np.array(train_h1_label)
    #############################################################################

    num_h0 = train_h0_label.sum()
    num_h1 = train_h1_label.sum()

    return train_h0_data, edge_feature_h0, adj_h0, edge_adj_h0, T_h0, train_h0_label, \
        train_h1_data, edge_feature_h1, adj_h1, edge_adj_h1, T_h1, train_h1_label, num_h0, num_h1, test_h0_label