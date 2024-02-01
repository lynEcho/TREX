import pandas as pd
import json
import argparse
import os
import sys
#from scipy.special import expit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
   
    args = parser.parse_args()
    dataset = args.dataset
    

    data_history = pd.read_csv(f'../../csvdata/{dataset}/{dataset}_train.csv')
    data_future = pd.read_csv(f'../../csvdata/{dataset}/{dataset}_test.csv')
    keyset_file = f'../../keyset/{dataset}_keyset.json'   
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    g_top_file = f'pop/{dataset}_pop.csv' #item_id,count
    g_top = pd.read_csv(g_top_file)
    g_top_list = g_top['item_id'].to_list()
    gtop_dict = dict(zip(g_top['item_id'], g_top['count']))    

    pred_dict = dict()
    rel_dict = dict()
    for user in keyset['test']:

        user_history = data_history[data_history['user_id'].isin([user])] #one user every time
        history_items = user_history['item_id'].tolist()
        s_pop_dict = dict() #{item: count}
        for item in history_items:
            if item not in s_pop_dict.keys():
                s_pop_dict[item] = 1
            else:
                s_pop_dict[item] += 1
        s_dict = sorted(s_pop_dict.items(), key=lambda d: d[1], reverse=True) #sort items via decreasing count, list [(0, 10), (4, 10), (6, 9), (8, 8), ...]
        
        pred = []
        rel = [0] * keyset['item_num']

        for item, cnt in s_dict:
            pred.append(item)
            rel[item] = cnt

        max_rel = max(rel)
        rel = [x / max_rel for x in rel] #first norm

        #add global popular
        ind = 0
        while(len(pred)<100):
            if g_top_list[ind] not in pred:
                pred.append(g_top_list[ind])
                rel[g_top_list[ind]] = gtop_dict[g_top_list[ind]] / g_top['count'].max() #second norm


            ind += 1
        pred_dict[user] = pred
        rel_dict[user] = rel

        #rel_dict[user] = expit(rel).tolist() 
        
    if not os.path.exists('gp_top_results/'):
        os.makedirs('gp_top_results/')
    with open(f'gp_top_results/{dataset}_pred0.json', 'w') as f:
        json.dump(pred_dict, f)
    with open(f'gp_top_results/{dataset}_rel0.json', 'w') as f:
        json.dump(rel_dict, f)


