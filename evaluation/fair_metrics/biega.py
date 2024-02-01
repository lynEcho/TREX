# %%
from sklearn import preprocessing
import numpy as np
import pandas as pd
min_max_scaler = preprocessing.MinMaxScaler()
import sys

# %%

def unfairness(rec_df, or_df, group, weight_vector):
    """
    IAA (Inequity of Amortized Attention).

    Related Paper: "Amortizing Individual Fairness in Rankings"
    First Author:  Asia J. Biega

    Args:
        rec_df (pandas.DataFrame): complete set of rankings to measure
        or_df: relevance score 
        group (GroupInfo Object): contains group information from rec_df
        weight_vector (position based object): see position module

    Columns Used
        rating -- 0/1, whether a user clicked on an item (aka relevance)
    """
    
    nlists = rec_df['user'].nunique() #number of users
    
    # group to compute per-group exposure and relevance
    g_att = weight_vector.groupby(rec_df[group.category]).sum() / nlists
    g_rel = or_df['score'].groupby(or_df[group.category]).sum() / nlists
   
    #g_rel = or_df[group.category].sum() / nlists
    #print(g_rel)

   

#     norm_att = min_max_scaler.fit_transform(np.array(g_att).reshape(-1,1))
#     #print(norm_att)
#     g_att = pd.Series(norm_att[0])
#     #g_rel = rec_df['rating'].groupby(rec_df[group.category]).sum() / nlists
#     norm_rel = min_max_scaler.fit_transform(np.array(g_rel).reshape(-1,1))
#     g_rel = pd.Series(norm_rel[0])
    # the metric is the L1 norm
    return np.abs(g_att - g_rel).sum()
