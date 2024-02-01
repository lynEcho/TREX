import json
import sys
import os
from tqdm import tqdm
from utils.metric import evaluate
from utils.data_container import get_data_loader
from utils.load_config import get_attribute
from utils.util import convert_to_gpu
from train.train_main import create_model
from utils.util import load_model
from scipy.special import expit
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--number', type=int, default=0, help='x')
    parser.add_argument('--best_model_path', type=str, required=True)
    
    args = parser.parse_args()

    dataset = args.dataset
    number = args.number
    model_path = args.best_model_path

    history_path = f'../../jsondata/{dataset}_history.json'
    future_path = f'../../jsondata/{dataset}_future.json'
    keyset_path = f'../../keyset/{dataset}_keyset.json'
    if not os.path.exists('results/'):     
        os.mkdir('results/')
    pred_path = f'results/{dataset}_pred{number}.json'
    pred_rel_path = f'results/{dataset}_rel{number}.json'
   
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)

    model = create_model()
    model = load_model(model, model_path)

    data_loader = get_data_loader(history_path=history_path,
                                    future_path=future_path,
                                    keyset_path=keyset_path,
                                    data_type='test',
                                    batch_size=1,
                                    item_embedding_matrix=model.item_embedding)

    model.eval()

    pred_dict = dict()
    pred_rel_dict = dict()
    test_key = keyset['test']
    user_ind = 0
    for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                    tqdm(data_loader)):
        pred_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency) #possibility
      

        pred_list = pred_data.detach().squeeze(0).numpy().argsort()[::-1][:100].tolist() #descending order, relevance scores are from 0.75646794 to -16.226542
        pred_rel_list = expit(pred_data.detach().squeeze(0).numpy()).tolist() #sigmoid, 3887users, 13897items
        assert len(pred_rel_list) == keyset['item_num']
        
        pred_dict[test_key[user_ind]] = pred_list
        pred_rel_dict[test_key[user_ind]] = pred_rel_list
       
        user_ind += 1

    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    with open(pred_rel_path, 'w') as f:
        json.dump(pred_rel_dict, f)
