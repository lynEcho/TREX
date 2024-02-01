import pandas as pd
import numpy as np
import math
import sys
# Small constant for avoiding divide by zero
EPSILON = 1.0e-6


    
    #To use these metrics on recommendations, users are considered as sequences.
    #To use thse metrics on IR tasks, runs for same query are used as sequences.
    

#sum_utility/num_group
def utility(rec_df, group):
    """
    Utility

    Related Paper: "Fairness of Exposure in Rankings"
    Co-Authors:     Ashudeep Singh, Thorsten Joachims

    Args:
        rec_df (pandas.DataFrame): complete set of recommendations
        group (GroupInfo Object): contains group information from rec_df

    Columns Used:
        rating -- 0/1, whether a user clicked on an item (aka relevance)
        group  -- see GroupInfo.py
    """

    rec_df['rating'] = rec_df['rating'].fillna(0)
    total = rec_df.groupby(group.category)['rating'].sum()
    
    minor_freq = total[group.minor] / group.group_freqs[group.minor]
    major_freq = total[group.major] / group.group_freqs[group.major]

    #unknown_freq = total[group.unknown] / group.group_freqs[group.unknown]
    return pd.Series({group.major: major_freq, group.minor: minor_freq})

#sum_exposure/num_group
def weighted_exposure(rec_df, group, weight_vector):
    """
    Weighted Exposure

    Related Paper: "Fairness of Exposure in Rankings"
    Co-Authors:     Ashudeep Singh, Thorsten Joachims

    Args:
        rec_df (pandas.DataFrame): complete set of recommendations
        group (GroupInfo Object): contains group information from rec_df
        weight_vector (position based object): see position module

    Columns Used:
        user  -- user who was recommended a list of items
        group -- see GroupInfo.py
    """

    group_df = rec_df[[group.major, group.minor]]
    
    # since we're summing, this is effectively the sum of the concatenation
   
   
    exp = group_df.T @ weight_vector #sum_exposure for each group
    # +divided by num_item_group
  
    exp[group.minor] = exp[group.minor] / group.group_freqs[group.minor]
    exp[group.major] = exp[group.major] / group.group_freqs[group.major]
    
    
    try:
        nusers = rec_df['user'].nunique()
        return exp / nusers
    except:
        nseq = rec_df['sequence'].nunique()

        return exp / nseq

#sum_ctr/num_group
def discounted_gain(rec_df, group, weight_vector):
    """
    Discounted Gain

    Related Paper: "Fairness of Exposure in Rankings"
    Co-Authors:     Ashudeep Singh, Thorsten Joachims

    Args:
        rec_df (pandas.DataFrame): complete set of recommendations
        group (GroupInfo Object): contains group information from rec_df
        weight_vector (position based object): see position module

    Columns Used:
        user  -- user who was recommended a list of items
        group_discount -- rating divided by weighting at current rank
    """
   
    
    rec_df['group_discount'] = rec_df['rating'] * weight_vector 
    try:
        nusers = rec_df['user'].nunique()
        
        ctr = rec_df.groupby(group.category)['group_discount'].sum().reindex([group.minor,group.major], fill_value=0)
      
        ctr[group.minor] = ctr[group.minor] / group.group_freqs[group.minor]
        ctr[group.major] = ctr[group.major] / group.group_freqs[group.major]
    
        return ctr / nusers
    except:
        nseq = rec_df['sequence'].nunique()
        return rec_df.groupby(group.category)['group_discount'].sum() / nseq


def demographic_parity(rec_df, group, weight_vector):
    """
    Demographic Parity

    Related Paper: "Fairness of Exposure in Rankings"
    Co-Authors:     Ashudeep Singh, Thorsten Joachims

    Args:
        rec_df (pandas.DataFrame): complete set of recommendations
        group (GroupInfo Object): contains group information from rec_df
        weight_vector (position based object): see position module

    Columns Used:
        group -- see GroupInfo in measures.py
    """

    exp = weighted_exposure(rec_df, group, weight_vector)
    
    #print("exp_dp: ", exp)
 
    if exp[group.major]==0.0:
        print("no unpop group1")
    return math.log(exp[group.minor] + EPSILON) - math.log(exp[group.major] + EPSILON)


def exposed_utility_ratio(rec_df, test_set, group, weight_vector):
    """
    Exposed Utility Ratio

    Original Metric Name: Disparate Treatment Ratio

    Related Paper: "Fairness of Exposure in Rankings"
    Co-Authors:     Ashudeep Singh, Thorsten Joachims

    Args:
        rec_df (pandas.DataFrame): complete set of recommendations
        test_set (pandas.DataFrame): ratings set
        group (GroupInfo Object): contains group information from rec_df
        weight_vector (position based object): see position module

    Columns Used
        group -- see GroupInfo in measures.py
    """

    exp = weighted_exposure(rec_df, group, weight_vector)
    util = utility(test_set, group)
    ratios = exp / util
    
    if ratios[group.major]==0.0:
        print("no unpop group2")

    return math.log(ratios[group.minor] + EPSILON) - math.log(ratios[group.major] + EPSILON)


def realized_utility_ratio(rec_df, test_set, group, weight_vector):
    """
    Realized Utility Ratio

    Original Metric Name: Disparate Impact Ratio

    Related Paper: "Fairness of Exposure in Rankings"
    Co-Authors:     Ashudeep Singh, Thorsten Joachims

    Args:
        rec_df (pandas.DataFrame): complete set of recommendations
        test_set (pandas.DataFrame): ratings set
        group (GroupInfo Object): contains group information from rec_df
        weight_vector (position based object): see position module

    Columns Used
        group  -- see GroupInfo in measures.py
        rating -- 0/1, whether a user clicked on an item (aka relevance)
    """

    rec_df['rating'] = rec_df['rating'].fillna(0.0) 
    exp = discounted_gain(rec_df, group, weight_vector)

    util = utility(test_set, group)
    ratios = exp / util
   
#    if ratios[group.major][0]==0.0:
#        return ratios[group.minor][0]
    if ratios[group.major]==0.0:
        print(ratios[group.major])
        print("no unpop group3")
      
    return math.log(ratios[group.minor] + EPSILON) - math.log(ratios[group.major] + EPSILON)
