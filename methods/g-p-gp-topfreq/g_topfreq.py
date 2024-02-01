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
    
    keyset_file = f'../../keyset/{dataset}_keyset.json'   
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)


    g_top_file = f'pop/{dataset}_pop.csv'
    g_top = pd.read_csv(g_top_file)

    top_items = g_top.head(100)
    gtop_dict = dict(zip(top_items['item_id'], top_items['count']))


    pred_dict = dict()
    rel_dict = dict()

    #same rec. for each user

    pred = []
    rel = [0] * keyset['item_num']
    for item, cnt in gtop_dict.items():
        pred.append(item)
        rel[item] = cnt
    
    for user in keyset['test']:
        pred_dict[user] = pred
        max_rel = max(rel)
        rel_dict[user] = [x / max_rel for x in rel] #divided by the max value

        #rel_dict[user] = expit(rel).tolist() 


    if not os.path.exists('g_top_results/'):
        os.makedirs('g_top_results/')

    with open(f'g_top_results/{dataset}_pred0.json', 'w') as f:
        json.dump(pred_dict, f)
    with open(f'g_top_results/{dataset}_rel0.json', 'w') as f:
        json.dump(rel_dict, f)


