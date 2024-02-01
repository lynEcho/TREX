# from util import instacartParser, prediction, evaluation
# from NextBasketRecFramework import uwPopMat, upcf, ipcf
import argparse
import json
import numpy as np
from scipy import sparse
# import similaripy as sim
import os
from similarity import *
import sys
from metrics import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path the dataset (for now we only applied it to the instacart dataset)
    parser.add_argument('--dataset', default='instacart', help="")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--number', type=int)
    # Preprocessing args
    parser.add_argument('--item_threshold', default=10, type=int)
    parser.add_argument('--basket_threshold', default=2, type=int)
    parser.add_argument('--subdata', default=0.05, type=float)
    parser.add_argument('--verbose', default=True, type=bool)
    # Methods : {UWPop, UPCF, IPCF} with/out recency
    parser.add_argument('--method_name', default='UPCF')
    # Method's parameters
    parser.add_argument('--recency', default=10, type=int)
    parser.add_argument('--asymmetry', default=0.75, type=float)
    parser.add_argument('--locality', default=10, type=int)
    parser.add_argument('--top_k', default=100, type=int)

    args = parser.parse_args()
    # Get the prgm args
    seed = args.seed
    number = args.number
    np.random.seed(seed)

    dataset = args.dataset
    item_threshold, basket_threshold, subdata, verbose = args.item_threshold, args.basket_threshold, args.subdata, args.verbose
    method_name = args.method_name
    recency, alpha, q, topk = args.recency, args.asymmetry, args.locality, args.top_k

    # Print framework parameters
    print("method_name:", method_name)
    print("recency:", recency, '\nasymmetry:', alpha, "\nlocality:", q)
    print('=============')

    data_path = f'../../mergedata/{dataset}_merged.json'
    with open(data_path, 'r') as f:
        data = json.load(f)

    # user_num = np.max([int(i) for i in data.keys()]) + 1
    uid_map_dict = dict()
    user_num = 0
    for uid in data.keys(): #reset all users from 0
        uid_map_dict[uid] = user_num
        user_num += 1
    # user_num +=
    item_num = 0
    item_map_dict = dict()
    rev_item_map_dict = dict()
    cf_user_list = []
    cf_item_list = []
    rc_user_list = []
    rc_item_list = []
    rc_score_list = []

    for user, bask_seq in data.items():
        history_seq = bask_seq[:-1] #except the last 1
        # computer cf part

        for bask in history_seq:
            for item in bask:
                if item not in item_map_dict.keys():
                    item_map_dict[item] = item_num #reset the items from 0 
                    rev_item_map_dict[item_num] = item
                    item_num += 1
                # if item > item_num:
                #     item_num = item
                cf_user_list.append(uid_map_dict[user])
                cf_item_list.append(item)

        # compute recency part
        if len(history_seq)>recency:
            recent_seq = history_seq[-recency:]
        else:
            recent_seq = history_seq

        freq_dict = dict()
        for bask in recent_seq:
            for item_o in bask:
                # item_f = item_map_dict[item_o]
                item_f = item_o
                if item_f not in freq_dict.keys():
                    freq_dict[item_f] = 1
                else:
                    freq_dict[item_f] += 1
        for item_f in freq_dict.keys():
            rc_score = freq_dict[item_f]/float(len(recent_seq))
            rc_user_list.append(uid_map_dict[user])
            rc_item_list.append(item_f)
            rc_score_list.append(rc_score)
            # cf_user_list.append(uid_map_dict[user])
            # cf_item_list.append(item)
    
    item_num += 1 
    # print(user_num)
    # print(item_num)
    rc_matrix = sparse.coo_matrix((rc_score_list, (rc_user_list, rc_item_list)), shape=(user_num, item_num))
    user_item_matrix = sparse.coo_matrix((np.ones((len(cf_user_list),), dtype=int), (cf_user_list, cf_item_list)), shape=(user_num, item_num))
    # print(user_item_matrix.toarray())
    # user_item_matrix = sparse.csr_matrix(user_item_matrix)
    user_item_matrix = user_item_matrix.tocsc().T
    # print(user_item_matrix.shape)
    sim_cls = Compute_Similarity(user_item_matrix, shrink=1000, asymmetric_alpha=alpha, similarity='asymmetric', topK=topk)
    usersim = sim_cls.compute_similarity().T
    # usersim = sim.asymmetric_cosine(user_item_matrix, None, alpha, k=topk) ###?????
    # print(type(usersim))
    # user_reccomendation = usersim.power(q).dot(sparse.csr_matrix(rc_matrix))
    usersim = usersim.toarray()
    for i in range(user_num):
        usersim[i][i] = 1.0
    # for i in range(10):
    #     print(usersim[i+1][i-1])
    #     print(usersim[i-1][i+1])

    usersim = sparse.csr_matrix(usersim)
    user_reccomendation = np.matrix.dot(usersim.power(q).toarray(), sparse.csr_matrix(rc_matrix).toarray())
    # user_reccomendation = sim.dot_product(usersim.power(q), sparse.csr_matrix(rc_matrix), k=topk)
    # user_reccomendation = sparse.csr_matrix(rc_matrix).toarray()

    keyset_path = f'../../keyset/{dataset}_keyset.json'
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)
    test_uid = keyset['test']
    item_total = keyset['item_num']

    if not os.path.exists('results/'):
        os.makedirs('results/')
    pred_path = f'results/{dataset}_pred{number}.json'
    rel_path = f'results/{dataset}_rel{number}.json'
    
    pred_dict = dict()
    rel_dict = dict()
    for uid in test_uid:
        u_pred = user_reccomendation[uid_map_dict[uid]] #score

        norm_rel = [sigmoid(x) for x in u_pred.tolist()]
        assert all(0 <= i <= 1 for i in norm_rel)
        pred_list = u_pred.argsort()[::-1][:100]
        # pred_list_o = [rev_item_map_dict[item] for item in pred_list]
        # o_pred_list = [rev_item_map_dict[item] for item in pred_list]

        pred_dict[uid] = [int(i) for i in pred_list]
        rel_dict[uid] = norm_rel + ([0] * (item_total-len(norm_rel)))
        assert len(rel_dict[uid]) == item_total

    
    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    with open(rel_path, 'w') as f:
        json.dump(rel_dict, f)

    print('Done')
