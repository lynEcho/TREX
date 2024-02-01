import pandas as pd
import json
import argparse
import os
from collections import Counter
import numpy as np
import random
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    args = parser.parse_args()
    dataset = args.dataset
   
    g_top_file = f'pop/{dataset}_pop.csv' #item_id,count(descending)
    g_top = pd.read_csv(g_top_file)

    if dataset == 'instacart':
        threshold = 87
    elif dataset == 'dunnhumby':
        threshold = 33
    elif dataset == 'tafeng':
        threshold = 42

    # grouping process
    group_dict = {'pop': [], 'unpop': []}
    ind = 0
    while ind < len(g_top):

        if g_top.at[ind,'count'] > threshold:
            group_dict['pop'].append(int(g_top.at[ind,'item_id']))
        else:
            group_dict['unpop'].append(int(g_top.at[ind,'item_id']))
        ind += 1
    
    with open(f'group_results/{dataset}_group_purchase.json', 'w') as f:
        json.dump(group_dict, f)

    
