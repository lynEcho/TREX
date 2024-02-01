import numpy as np
import math
import sys
from numpy.core.fromnumeric import argsort

#note ground truth is vector, rank_list is the sorted item index.

def label2vec(label_list, input_size):
    #label_list -> list
    #input_size -> item number
    label_vec = np.zeros(input_size)
    for label in label_list:
        label_vec[label]=1
    return label_vec

def get_repeat_explore(repeat_list, pred_rank_list, k):
    count = 0
    repeat_cnt = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in repeat_list:
            repeat_cnt += 1
        count += 1
    repeat_ratio = repeat_cnt/k
    return repeat_ratio, 1-repeat_ratio

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


def get_precision_recall_Fscore(groundtruth, pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        #print('postivie is 0')
    else:
        precision = correct/positive
    if 0 == truth:
        recall = 0
        flag = 1
        #print('recall is 0')
    else:
        recall = correct/truth

    if flag == 0 and precision + recall > 0:
        F = 2*precision*recall/(precision+recall)
    else:
        F = 0
    return precision, recall, F, correct


def get_Fairness(exp0, exp1, exp2, u0, u1, u2):   

    DTR01 = abs((exp0/u0)/(exp1/u1) - 1)
    DTR02 = abs((exp0/u0)/(exp2/u2) - 1)
    DTR12 = abs((exp1/u1)/(exp2/u2) - 1)
    fairness = (DTR01 + DTR02 + DTR12)/3
    return fairness

'''
def get_Exposure(pred_list, item_group, k):

    count = 0
    exp0 = 0
    exp1 = 0
    exp2 = 0
    for pred in pred_list:
        if count >= k:
            break
        if pred in item_group['group0']:
            exp0 += (1)/math.log2(count + 1 + 1)
        elif pred in item_group['group1']:
            exp1 += (1)/math.log2(count+1+1)
        elif pred in item_group['group2']:
            exp2 += (1)/math.log2(count+1+1)
        count += 1
    return exp0, exp1, exp2

'''
def get_Exposure(pred_list, item_group, k):

    count = 0
    exp0 = 0
    exp1 = 0
    exp2 = 0
    for pred in pred_list:
        if count >= k:
            break
        if pred in item_group['group0']:
            exp0 += 1
        elif pred in item_group['group1']:
            exp1 += 1
        else:
            exp2 += 1
        count += 1
    return exp0, exp1, exp2


'''
def get_Utility(item_group, item_count, user):

    u0 = 0
    u1 = 0
    u2 = 0
    for item in item_count[user].keys():
        if int(item) in item_group['group0']:
            u0 += item_count[user][item]      
        elif int(item) in item_group['group1']:
            u1 += item_count[user][item]      
        elif int(item) in item_group['group2']:
            u2 += item_count[user][item]  
    return u0, u1, u2
'''

def get_Utility(item_group, truth_list, pred_rank_list, k):
    count = 0
    u0 = 0
    u1 = 0
    u2 = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            if pred in item_group['group0']:
                u0 += 1      
            elif pred in item_group['group1']:
                u1 += 1     
            elif pred in item_group['group2']:
                u2 += 1
        count += 1
    return u0, u1, u2



def get_Unfair(truth, pred, size, att, rel):
    
    for pos, index in enumerate(pred[:size]): #0, 2579
        att[index] = att[index] + pos_bias(pos+1, size, 0.5)
        #att[index] = att[index] + 1
        if index in truth:
            rel[index] = rel[index] + 1

    return att, rel

def pos_bias(pos: int, k: int, prob: float) -> float:
    """
    Receives a position and calculates the bias of said position. It will calculate
    only for the first k subjects, after k, the bias will be 0.

    Parameters
    ----------
    pos : integer
        Position
    k : integer
        Amount of top subjects to consider for calculations
    prob : float
        Probability that any subject will be chosen
    """
    if pos > k:
        return 0
    else:
        return prob * pow(1-prob, pos-1)
