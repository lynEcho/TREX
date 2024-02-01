import json
import argparse
import pandas as pd
import numpy as np
from metrics import *
import os


#explore list depends on filtered repeat list

def gen_basket_threshold(rep_pred, keyset, v, item_cate_dict, global_item, data_history, type):
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
        
        rep_cate = [] #category of 'user_pred'    
        for i in user_pred:
            if item_cate_dict[i] not in rep_cate:
                rep_cate.append(item_cate_dict[i])

        user_history = data_history[data_history['user_id'].isin([int(user)])]
        history_items = user_history['item_id'].tolist()


        for i in global_item:
            if i not in history_items: #new item
                if item_cate_dict[i] not in rep_cate:
                    user_pred.append(i)
                    rep_cate.append(item_cate_dict[i])

                    if len(user_pred) >= 20:
                        break

        
        data_pred[user] = user_pred #final basket
    return data_pred

#generate explore list and final basket
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

    cate_file = f'category/{dataset}_group_category.json'
    with open(cate_file, 'r') as f:
        category = json.load(f)

    item_cate_dict = dict()
    for key, value in category.items():
        for item in value:
            item_cate_dict[item] = key


    item_freq_file = f'popularity/{dataset}_frequency.csv'
    item_freq = pd.read_csv(item_freq_file)
    global_item = item_freq['item_id'].tolist()

    data_history = pd.read_csv(f'csvdata/{dataset}/{dataset}_train.csv')


    with open(rep_pred_file, 'r') as f:
        rep_pred = json.load(f)

    if not os.path.exists('final_results_div'):
        os.makedirs('final_results_div')
    
    for v in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:

        test_data_pred = gen_basket_threshold(rep_pred, keyset, v, item_cate_dict, global_item, data_history, type='test')
        with open('final_results_div/'+dataset+'_pred_'+str(v)+'.json', 'w') as f:
            json.dump(test_data_pred, f)


    return v


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="instacart")
    args = parser.parse_args()
    dataset = args.dataset

    size = 10

    eval_basket(dataset, size)

