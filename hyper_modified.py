import os
import sys 
sys.path.append("..") 
import json
import argparse
import numpy as np
from metrics import *


def get_repeat_pred(dataset, size, mode):
    history_file = 'trex-framework/jsondata/' + dataset + '_history.json'
    truth_file = 'trex-framework/jsondata/' + dataset + '_future.json'
    utility = []
    for alpha in np.arange(0, 1.01, 0.1):
        
        keyset_file = 'trex-framework/keyset/' + dataset + '_keyset.json'

        with open(history_file, 'r') as f:
            history_data = json.load(f)
        with open(truth_file, 'r') as f:
            data_truth = json.load(f)
        with open(keyset_file, 'r') as f:
            keyset = json.load(f)

        item_num = keyset['item_num'] #all items(train+val+test)+1
        train_user = keyset['train']

        #generate repetition score
        total = np.zeros(item_num) #users who bought item i at least once
        rep = np.zeros(item_num) 
        rep_cnt = np.zeros(item_num) 
        for item_ind in range(1, item_num): #for each item (item_id starts from 1).
            for user in train_user: #for each user
                cnt = 0.0 #frequency for each user
                for bask in history_data[user][1:-1]: #all history baskets
                    if item_ind in bask:
                        cnt += 1
                if cnt != 0:
                    total[item_ind] += 1 #users who bought item i at least once
                    if cnt >= 2:
                        rep[item_ind] += 1 #users who repurchase item i 
                        rep_cnt[item_ind] += math.pow(cnt-1, alpha) #each user contributes to rep level

        if mode == 'no_item':
            rep_prob = np.ones(item_num)

        if mode == 'cnt_item':
            rep_prob = np.zeros(item_num) 
            for item_ind in range(1, item_num):
                if rep[item_ind] == 0:
                    rep_prob[item_ind] = 0.0
                else:
                    rep_prob[item_ind] = rep_cnt[item_ind]/total[item_ind] #equ5

            avg_rep_prob = np.mean(rep_prob)
            for item_ind in range(1, item_num):
                rep_prob[item_ind] = rep_prob[item_ind] + avg_rep_prob/(total[item_ind]+1) #equ6, +1 for "total[item_ind] = 0"
        
        # search param on val, beta
        for beta in np.arange(0, 1.01, 0.1): 
        
            val_user = keyset['val']
            pred_val_dict = dict()
            for user in val_user: #for each user
                interests = np.zeros(item_num)
                user_data = history_data[user][1:-1][::-1] #reverse, from now to past
                bask_len = len(user_data)
                for rep_ind in range(bask_len):
                    for item in user_data[rep_ind]:
                        interests[item] += math.pow(beta, rep_ind)
                user_pred = interests * rep_prob
                # user_pred = interests
                pred = np.argsort(user_pred)[::-1] #descending order
                pred_item = [int(item) for item in pred if user_pred[item] > 0] #repeat item list
                pred_score = [user_pred[ind] for ind in pred_item]
                pred_val_dict[user] = [pred_item, pred_score]

                
            recall_list = []
            for user in val_user:
                pred = pred_val_dict[user][0]
                truth = data_truth[user][1]
                u_recall = get_Recall(truth, pred, size)
                recall_list.append(u_recall)
            val_recall = np.mean(recall_list)

            

            # get predict
            test_user = keyset['test']
            pred_test_dict = dict()
            for user in test_user:
                interests = np.zeros(item_num)
                user_data = history_data[user][1:-1][::-1]
                bask_len = len(user_data)
                for rep_ind in range(bask_len):
                    for item in user_data[rep_ind]:
                        interests[item] += math.pow(beta, rep_ind) #equ7
                user_pred = interests * rep_prob #equ3
                # user_pred = interests
                pred = np.argsort(user_pred)[::-1]
                pred_item = [int(item) for item in pred if user_pred[item] > 0]
                pred_score = [user_pred[ind] for ind in pred_item]
                pred_test_dict[user] = [pred_item, pred_score]

            recall_list = []
            for user in test_user:
                pred = pred_test_dict[user][0]
                truth = data_truth[user][1]
                u_recall = get_Recall(truth, pred, size)
                recall_list.append(u_recall)
            test_recall = np.mean(recall_list)

            utility.append([alpha, beta, val_recall, test_recall])

            pred_file = mode+'/'+ dataset + '_pred-'+str(size)+'-'+str(alpha)+'-'+str(beta)+'.json'
            with open(pred_file, 'w') as f:
                json.dump(pred_test_dict, f)

    with open(f'trex-framework/cnt_item/val_test_'+ dataset +'.txt', 'w') as f:
        for line in utility:
            f.write(str(line))
            f.write('\n')  




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="instacart")
    args = parser.parse_args()
    dataset = args.dataset
    mode = 'cnt_item'
    size = 10
    get_repeat_pred(dataset, size, mode)


