# this file is to genenrate candidatse.
import numpy as np

def get_repeat_candidates(data_train):
    repeat = []
    for basket_seq in data_train:
        user_item = []
        for basket in basket_seq:
            for item in basket:
                if item not in user_item:
                    user_item.append(item)
        repeat.append(user_item)
    return repeat

def get_explore_candidates(data_train, total_count):
    explore = []
    total_basket = [item for item in range(total_count)]
    for basket_seq in data_train:
        explore_item = total_basket.copy()
        for basket in basket_seq:
            for item in basket:
                if item in explore_item:
                    explore_item.remove(item)
        explore.append(explore_item)
    return explore

def get_item_candidates(data_train, total_count, candidates_count):
    pass

def get_user_candidates(data_train, total_count, candidates_count):
    pass

def get_popular_candidates(data_train, user_sum, total_count, candidates_count):
    #here might be soem problem, if the test data should be counted in the popular items, I think yes here.
    item_freq = np.zeros(total_count)
    for basket_seq in data_train:
        for basket in basket_seq:
            for item in basket:
                item_freq[item] +=1
    pop_candidates = item_freq.argsort()[::-1][:candidates_count]
    pop = []
    for i in range(user_sum):
        pop.append(list(pop_candidates))
    return pop