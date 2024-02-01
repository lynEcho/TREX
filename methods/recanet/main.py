import pandas as pd
import sys
from models.mlp_v12 import MLPv12
from utils.metrics import *
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='ReCANet')
    parser.add_argument('-dataset', type=str, default='instacart')
    parser.add_argument('-user_embed_size', type=int, default=32)
    parser.add_argument('-item_embed_size', type=int, default=128)
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-history_len', type=int, default=20)
    parser.add_argument('-number', type=int, default=0)
    parser.add_argument('-job_id', type=int, default=0)
    parser.add_argument('-seed_value', type=int, default=12321)
    args = parser.parse_args()
    return args


args = parse_args()


seed_value = args.seed_value
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

dataset = args.dataset



data_path = f'../../csvdata/{args.dataset}/'


print(dataset)
train_baskets = pd.read_csv(data_path+dataset+f'_train.csv') 
test_baskets = pd.read_csv(data_path+dataset+f'_test.csv')
valid_baskets = pd.read_csv(data_path+dataset+f'_valid.csv')

print('data read')
user_test_baskets_df = test_baskets.groupby('user_id')['item_id'].apply(list).reset_index()
user_test_baskets_dict = dict(zip( user_test_baskets_df['user_id'],user_test_baskets_df['item_id'])) #{user:items}

model = MLPv12(train_baskets, test_baskets,valid_baskets,data_path, args.user_embed_size,args.item_embed_size,64,64,64,64,64,args.history_len, job_id = args.job_id,seed_value=args.seed_value)

model.train()
print('model trained')

user_predictions, user_rel = model.predict()
if not os.path.exists('results/'):     
        os.mkdir('results/')
pred_path = f'results/{args.dataset}_pred{args.number}.json'
pred_rel_path = f'results/{args.dataset}_rel{args.number}.json'

with open(pred_path, 'w') as f:
    json.dump(user_predictions, f)
with open(pred_rel_path, 'w') as f:
    json.dump(user_rel, f)

#below is evaluation
'''
final_users = set(model.test_users).intersection(set(list(user_test_baskets_dict.keys())))
print('predictions ready',len(user_predictions))
print('number of final test users:',len(final_users))
for k in [5,10,20,'B']:
    print(k)
    recall_scores = {}
    ndcg_scores = {}
    zero = 0
    for user in final_users:

        top_items = []
        if user in user_predictions:
            top_items = user_predictions[user]
        else:
            zero+=1

        if k == 'B':
            recall_scores[user] = recall_k(user_test_baskets_dict[user],top_items,len(user_test_baskets_dict[user]))
            ndcg_scores[user] = ndcg_k(user_test_baskets_dict[user],top_items,len(user_test_baskets_dict[user]))
        else:
            recall_scores[user] = recall_k(user_test_baskets_dict[user],top_items,k)
            ndcg_scores[user] = ndcg_k(user_test_baskets_dict[user],top_items,k)
    #print(zero)
    print('recall:',np.mean(list(recall_scores.values())))
    print('ndcg:',np.mean(list(ndcg_scores.values())))
'''
