import json
import argparse
import random
import numpy as np
import pandas as pd
import os



def generate_sampled_list(choices, probabilities, size):

    # Use random.choices to generate a sampled list based on probabilities
    sampled_list = random.choices(choices, probabilities, k=size)

    return sampled_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="instacart")
    args = parser.parse_args()
    dataset = args.dataset


    data_history = pd.read_csv(f'csvdata/{dataset}/{dataset}_train.csv')

    keyset_file = f'keyset/{dataset}_keyset.json'   
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    item_freq_file = f'popularity/{dataset}_frequency.csv'
    item_freq = pd.read_csv(item_freq_file)

    group_file = f'popularity/{dataset}_group_purchase.json'   
    with open(group_file, 'r') as f:
        group = json.load(f)

    #filter out the popular items

    filtered_item_freq = item_freq[~item_freq['item_id'].isin(group['pop'])]

    #compute probability based on frequency
    
    total_count = filtered_item_freq['count'].sum()

    filtered_item_freq['probability'] = filtered_item_freq['count'] / total_count

    items = filtered_item_freq['item_id'].tolist()
    probabilities = filtered_item_freq['probability'].tolist()

    #sampling

    list_size = 100

    test_user = keyset['train']
    pred_test_dict = dict()
    for user in test_user:
        
        sampled_list = generate_sampled_list(items, probabilities, list_size)

        #delete duplicate items
        unique_list = list(set(sampled_list))
        
        user_history = data_history[data_history['user_id'].isin([int(user)])]

        history_items = user_history['item_id'].tolist()
        
        #delete repeat items for each user
        filter_list = [item for item in unique_list if item not in history_items]
        
        assert len(filter_list) >= 20

        pred_test_dict[user] = filter_list
    
    if not os.path.exists('ex-fair_result'):
        os.makedirs('ex-fair_result')

    pred_file = f'ex-fair_result/{dataset}_pred.json'
    with open(pred_file, 'w') as f:
        json.dump(pred_test_dict, f)

    

