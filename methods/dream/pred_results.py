import json
import glob
from Explainablebasket import *
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    
    parser.add_argument('--attention', type=int, default=0)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--number', type=int)

    args = parser.parse_args()
    dataset = args.dataset
    
    attention = args.attention
    number = args.number
    seed = args.seed
    
    random.seed(seed)
    torch.manual_seed(seed)

    history_file = '../../jsondata/'+dataset+'_history.json'
    future_file = '../../jsondata/'+dataset+'_future.json'

    with open(history_file, 'r') as f:
        data_history = json.load(f)
    with open(future_file, 'r') as f:
        data_future = json.load(f)
    with open(dataset+'conf.json', 'r') as f:
        conf = json.load(f)
    #for mode in ['attention']:
    for mode in str(attention):
        conf['attention'] = mode
        conf['loss_mode'] = 0  # bceloss
        para_path = glob.glob('./models/'+dataset+'/*')
        
        keyset_file = '../../keyset/'+dataset+'_keyset'+'.json'

        if not os.path.exists('results/'):
            os.makedirs('results/')

        pred_file = 'results/'+dataset+'_pred'+str(number)+'.json'
        rel_file = 'results/'+dataset+'_rel'+str(number)+'.json'


        with open(keyset_file, 'r') as f:
            keyset = json.load(f)
        conf['item_num'] = keyset['item_num']
        conf['device'] = torch.device("cpu")
        keyset_test = keyset['test']

        checkpoint_file = []
        for path in para_path:

            path_l = path.split('-')
           
            if path_l[4] == mode and path_l[3] == str(seed):
                checkpoint_file.append(path)


        model = NBRNet(conf, keyset)
        checkpoint = torch.load(checkpoint_file[0], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
        print(message_output)
        model.eval()
        pred_dict = dict()
        rel_dict = dict()
        for user in keyset_test:
            basket = [data_history[user][1:-1]]
            cand = [[item for item in range(keyset['item_num'])]]
            scores = model.forward(basket, cand)
            rel_dict[user] = scores[0].detach().numpy().tolist()
            assert all(0 <= i <= 1 for i in rel_dict[user])
            pred_dict[user] = scores[0].detach().numpy().argsort()[::-1][:100].tolist()
       
        with open(pred_file, 'w') as f:
            json.dump(pred_dict, f)

        with open(rel_file, 'w') as f:
            json.dump(rel_dict, f)
