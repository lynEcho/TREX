import json
import argparse
import pandas as pd
import numpy as np
from metrics import *
import os

def gen_basket_threshold(rep_pred, expl_pred, keyset, v, type):
    # v is the threshold
    data_pred = dict()
    rep_cnt_dict = dict()

    for user in keyset[type]:
        rep_cnt = 0 #space for rep items
        for rep_score in rep_pred[user][1]:
            if rep_score >= v:
                rep_cnt += 1
        rep_cnt_dict[user] = rep_cnt

    for user in keyset[type]:
        rep_cnt = rep_cnt_dict[user]
        user_pred = rep_pred[user][0][:rep_cnt] #filter rep items   
        for item in expl_pred[user]:   
            if len(user_pred) > 20: #max size of basket
                break
            if item not in user_pred:
                user_pred.append(item)
        data_pred[user] = user_pred #final basket
    return data_pred

def get_basket_eval(data_pred, data_truth, data_history, keyset, size, type):
    ndcg = []
    recall = []
    hit = []
    repeat_ratio = []
    explore_ratio = []
    recall_repeat = []
    recall_explore = []
    hit_repeat = []
    hit_explore = []

    for user in keyset[type]:
        pred = data_pred[user]
        truth = data_truth[user][1]

        user_history = data_history[data_history['user_id'].isin([int(user)])]
        repeat_items = list(set(user_history['item_id']))
    
        truth_repeat = list(set(truth) & set(repeat_items))  # might be none
        truth_explore = list(set(truth) - set(truth_repeat))  # might be none

        u_ndcg = get_NDCG(truth, pred, size)
        ndcg.append(u_ndcg)
        u_recall = get_Recall(truth, pred, size)
        recall.append(u_recall)
        u_hit = get_HT(truth, pred, size)
        hit.append(u_hit)

        u_repeat_ratio, u_explore_ratio = get_repeat_explore(repeat_items, pred, size)  # here repeat items
        repeat_ratio.append(u_repeat_ratio)
        explore_ratio.append(u_explore_ratio)

        if len(truth_repeat) > 0:
            u_recall_repeat = get_Recall(truth_repeat, pred,
                                         size)  # here repeat truth, since repeat items might not in the groundtruth
            recall_repeat.append(u_recall_repeat)
            u_hit_repeat = get_HT(truth_repeat, pred, size)
            hit_repeat.append(u_hit_repeat)

        if len(truth_explore) > 0:
            u_recall_explore = get_Recall(truth_explore, pred, size)
            u_hit_explore = get_HT(truth_explore, pred, size)
            recall_explore.append(u_recall_explore)
            hit_explore.append(u_hit_explore)
    return  np.mean(recall), np.mean(ndcg), np.mean(hit), np.mean(repeat_ratio), np.mean(explore_ratio)\
        ,np.mean(recall_repeat), np.mean(recall_explore), np.mean(hit_repeat), np.mean(hit_explore)

def eval_basket(dataset, size):
    
    history_file = f'csvdata/{dataset}/{dataset}_train.csv'
    keyset_file = f'keyset/{dataset}_keyset.json'
    truth_file = f'jsondata/{dataset}_future.json'

    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)
    data_history = pd.read_csv(history_file)

    rep_pred_file = f'repeat_result/{dataset}_pred.json'
    exp_pred_file = f'ex-fair_result/{dataset}_pred.json'

    with open(rep_pred_file, 'r') as f:
        rep_pred = json.load(f)
    with open(exp_pred_file, 'r') as f:
        expl_pred = json.load(f)
    '''
    best_val_recall = 0
    best_val_hit = 0

    threshold = 0.0
    for v in np.arange(0.0, 0.5, 0.02):
        data_pred = gen_basket_threshold(rep_pred, expl_pred, keyset, v, type='val')  #generate final basket {user:final basket}

        val_recall, _, val_hit, _, _, _, _, _, _ = \
            get_basket_eval(data_pred, data_truth, data_history, keyset, size, type='val') #evaluate

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_val_hit = val_hit
            threshold = v
    print('Val threshold:', threshold)
    print('Val performance:', best_val_recall, best_val_hit)
    '''
    if not os.path.exists('final_results_fair'):
        os.makedirs('final_results_fair')

    for v in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
    
        test_data_pred = gen_basket_threshold(rep_pred, expl_pred, keyset, v, type='test')
        with open('final_results_fair/'+dataset+'_pred_'+str(v)+'.json', 'w') as f:
            json.dump(test_data_pred, f)

    return v

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="instacart")
    args = parser.parse_args()
    dataset = args.dataset

    size = 10
    eval_basket(dataset, size)
    

