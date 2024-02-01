import math

def get_DCG(truth_list, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            dcg += (1)/math.log2(count+1+1)
        count += 1
    return dcg

def get_NDCG(truth_list, pred_rank_list, k):
    dcg = get_DCG(truth_list, pred_rank_list, k)
    idcg = 0
    num_item = len(truth_list)
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg

def get_HT(truth_list, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            return 1
        count += 1
    return 0

def get_Recall(truth_list, pred_rank_list, k):
    truth_num = len(truth_list)
    count = 0
    correct = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            correct += 1
        count += 1
    recall = correct/truth_num
    return recall


