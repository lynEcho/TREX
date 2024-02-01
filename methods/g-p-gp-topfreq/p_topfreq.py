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

    pred_dict = dict()
    pred_rel_dict = dict()
   
    for user, user_data in data_future.groupby('user_id'):
        
        user_history = data_history[data_history['user_id'].isin([user])]
       
        history_items = user_history['item_id'].tolist()
        
        # print(history_items)
        s_pop_dict = dict()
        for item in history_items:
            if item not in s_pop_dict.keys():
                s_pop_dict[item] = 1
            else:
                s_pop_dict[item] += 1
        s_dict = sorted(s_pop_dict.items(), key=lambda d: d[1], reverse=True) #[(0, 10), (4, 10), (6, 9), (8, 8)...]
        
        pred = []
        rel = [0] * keyset['item_num']
        for item, cnt in s_dict:
            pred.append(item)
            rel[item] = cnt

        pred_dict[user] = pred
        #pred_rel_dict[user] = expit(rel).tolist() 
        
        max_rel = max(rel)
        pred_rel_dict[user] = [x / max_rel for x in rel] #divided by the max value
    
   

    if not os.path.exists('p_top_results/'):
        os.makedirs('p_top_results/')

    with open(f'p_top_results/{dataset}_pred0.json', 'w') as f:
        json.dump(pred_dict, f)

    with open(f'p_top_results/{dataset}_rel0.json', 'w') as f:
        json.dump(pred_rel_dict, f)



