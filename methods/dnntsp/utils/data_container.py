import json
from collections import defaultdict
import dgl
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
from sklearn.preprocessing import normalize
import sys
sys.path.append("..")
from dnntsp.utils.util import convert_to_gpu
from dnntsp.utils.util import get_truth_data
from dnntsp.utils.load_config import get_attribute


class SetDataset(Dataset):
    def __init__(self, history_path, future_path, keyset_path, item_embedding_matrix, key=None):
        """
        Args:
            data_path: str
            key: str
        """
        with open(history_path, 'r') as f:
            data_history = json.load(f)
        with open(future_path, 'r') as f:
            data_future = json.load(f)
        with open(keyset_path, 'r') as f:
            data_keyset = json.load(f)

        self.data_list = []
        self.item_embedding_matrix = item_embedding_matrix

        if key is None:
            print('it will not happen')
            for key in ['train', 'val', 'test']:
                user_list = data_keyset[key]
                for user in user_list:
                    h_basket = data_history[user][1:-1]
                    f_basket = data_future[user][1]
                    h_basket.append(f_basket)
                    user_data = [torch.tensor(list(set(basket))) for basket in h_basket]
                    self.data_list.append(user_data)
        elif key == 'train':
            user_list = data_keyset[key]
            for user in user_list:
                h_basket = data_history[user][1:-1]
                user_data = [torch.tensor(list(set(basket))) for basket in h_basket]
                self.data_list.append(user_data)
        else:
            user_list = data_keyset[key]
            for user in user_list:
                h_basket = data_history[user][1:-1]
                f_basket = data_future[user][1]
                h_basket.append(f_basket)
                user_data = [torch.tensor(list(set(basket))) for basket in h_basket]
                self.data_list.append(user_data)
        
    def __getitem__(self, index):
        """
        :param index:
        :return:  g, graph, fully connected, containing N nodes, unweighted
                  nodes_feature, tensor  (N, item_embedding)
                  edges_weight, tensor (T, N*N)
                  nodes, tensor (N, )
                  user_data, list, (baskets, items)
        """
        # list of tensors
        user_data = self.data_list[index]
        #print(index)
        #print(user_data)
        # nodes -> tensor,  len(nodes) = N
        # may change the order of appearing items in dataset
        nodes = self.get_nodes(baskets=user_data[:-1]) #except the last1 basket
        #print(nodes)
        # N * item_embedding tensor
        
        nodes_feature = self.item_embedding_matrix(convert_to_gpu(nodes.long())) #use this for training
        #nodes_feature = self.item_embedding_matrix(nodes.long()) #use this for prediction


        # construct graph for the user
        project_nodes = torch.tensor(list(range(nodes.shape[0])))
        # construct fully connected graph, containing N nodes, unweighted
        # (0, 0), (0, 1), ..., (0, N-1), (1, 0), (1, 1), ..., (1, N-1), ...
        # src -> [0, 0, 0, ... N-1, N-1, N-1, ...],  dst -> [0, 1, ..., N-1, ..., 0, 1, ..., N-1]
        src = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=1).flatten().tolist()
        dst = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=0).flatten().tolist()
        g = dgl.graph((src, dst), num_nodes=project_nodes.shape[0])
        edges_weight_dict = self.get_edges_weight(user_data[:-1])
        # add self-loop
        for node in nodes.tolist():
            if edges_weight_dict[(node, node)] == 0.0:
                edges_weight_dict[(node, node)] = 1.0
        # normalize weight
        max_weight = max(edges_weight_dict.values())
        for i, j in edges_weight_dict.items():
            edges_weight_dict[i] = j / max_weight
        # get edge weight for each timestamp, shape (T, N*N)
        # print(edges_weight_dict)
        edges_weight = []
        for basket in user_data[:-1]:
            basket = basket.tolist()
            # list containing N * N weights of elements
            edge_weight = []
            for node_1 in nodes.tolist():
                for node_2 in nodes.tolist():
                    if (node_1 in basket and node_2 in basket) or (node_1 == node_2):
                        # each node has a self connection
                        edge_weight.append(edges_weight_dict[(node_1, node_2)])
                    else:
                        edge_weight.append(0.0)
            edges_weight.append(torch.Tensor(edge_weight))
        # tensor -> shape (T, N*N)
        edges_weight = torch.stack(edges_weight)
        return g, nodes_feature, edges_weight, nodes, user_data #here

    def __len__(self):
        return len(self.data_list)

    def get_nodes(self, baskets):
        """
        get items in baskets
        :param baskets:  list (baskets_num, items_num)  each element is a tensor
        :return: tensor ([item_1, item_2, ... ])
        """
        # convert tensor to int
        baskets = [basket.tolist() for basket in baskets]
        items = torch.tensor(list(set(itertools.chain.from_iterable(baskets))))
        return items

    def get_edges_weight(self, baskets):
        """
        count the appearing counts of items in baskets
        :param baskets:  list (baskets_num, items_num)  each element is a tensor
        :return: dict, each item, key -> (n_1, n_2),  value -> weight.
        or if edge has features, then value -> {"weight":edge_weight, "features":...}
        """
        # convert tensor to int
        edges_weight_dict = defaultdict(float)
        for basket in baskets:
            basket = basket.tolist()
            for i in range(len(basket)):
                for j in range(i + 1, len(basket)):
                    edges_weight_dict[(basket[i], basket[j])] += 1.0
                    edges_weight_dict[(basket[j], basket[i])] += 1.0
        return edges_weight_dict


def collate_set_across_user(batch_data):
    """
    Args:
        batch_data: list, shape (batch_size, XXX)

    Returns:
        graph:
        train_data: list, shape (batch_size, baskets_num - 1, items_num)
        truth_data: list of tensors, shape (batch_size, items_total) or (batch_size, baskets_num - 1, items_total)
    """
    # g, nodes_feature, edges_weight, nodes, user_data
    # zip * -> unpack
    ret = list()
    for idx, item in enumerate(zip(*batch_data)):
        # assert type(item) == tuple
        if isinstance(item[0], dgl.DGLGraph):
            ret.append(dgl.batch(item))
        elif isinstance(item[0], torch.Tensor):
            if idx == 2:
                # pad edges_weight sequence in time dimension batch, (T, N*N)
                # (T_max, N*N)
                max_length = max([data.shape[0] for data in item])
                edges_weight, lengths = list(), list()
                for data in item:
                    if max_length != data.shape[0]:
                        edges_weight.append(torch.cat((data, torch.stack(
                            [torch.eye(int(data.shape[1] ** 0.5)).flatten() for _ in range(max_length - data.shape[0])],
                            dim=0)), dim=0))
                    else:
                        edges_weight.append(data)
                    lengths.append(data.shape[0])
                # (T_max, N_1*N_1 + N_2*N_2 + ... + N_b*N_b)
                ret.append(torch.cat(edges_weight, dim=1))
                # (batch, )
                ret.append(torch.tensor(lengths))
            else:
                # nodes_feature -> (N_1 + N_2, .. + N_b, item_embedding) or nodes -> (N_1 + N_2, .. + N_b, )
                ret.append(torch.cat(item, dim=0))
        elif isinstance(item[0], list):
            data_list = item
        else:
            raise ValueError(f'batch must contain tensors or graphs; found {type(item[0])}')

    truth_data = get_truth_data([dt[-1] for dt in data_list])
    ret.append(truth_data)

    # tensor (batch, items_total), for frequency calculation
    users_frequency = np.zeros([len(batch_data), get_attribute('items_total')])
    for idx, baskets in enumerate(data_list):
        for basket in baskets:
            for item in basket:
                users_frequency[idx, item] = users_frequency[idx, item] + 1
    users_frequency = normalize(users_frequency, axis=1, norm='max')
    ret.append(torch.Tensor(users_frequency))

    # (g, nodes_feature, edges_weight, lengths, nodes, truth_data, individual_frequency)
    return tuple(ret)


def get_data_loader(history_path, future_path, keyset_path, data_type, batch_size, item_embedding_matrix):
    """
    Args:
        data_path: str
        data_type: str, 'train'/'validate'/'test'
        batch_size: int
    Returns:
        data_loader: DataLoader
    """

    dataset = SetDataset(history_path, future_path, keyset_path, item_embedding_matrix=item_embedding_matrix, key=data_type)
    print(f'{data_type} data length -> {len(dataset)}')
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=collate_set_across_user)
    return data_loader
