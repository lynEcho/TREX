import numpy as np
import pandas as pd
import sys
class rbp:
    """
    Rank-Biased Precision

    Args:
        rec_df (pandas.DataFrame): set of recommendations (single list or complete)
        patience (float): controls how fast metric values may fall
                          as rank increases
    """
    def __init__(self, patience=0.5):
        self.patience = patience

    def __call__(self, rec_df):
        # fix for 1-based ranking
        return self.patience ** (rec_df['rank'] - 1)

    def ideal(self, utilities):
        """
        ideal weighting

        Args:
            utilities (pandas.Series):

        Returns:
            numpy array: array with same size as utlities
        """
        # equally distribute exposure
        m = len(utilities) #length of ground truth
        num = 1 - self.patience ** m
        den = m * (1 - self.patience)
        return np.full(m, num / den)


class cascade:
    def __init__(self, patience=0.5, stop=0.5):
        self.patience = patience
        self.stop = stop

    def __call__(self, rec_df):
        left = self.patience ** (rec_df['rank'] - 1)
        right = 1 - (rec_df['rating'] * self.stop)

        return left * right.cumprod()

    def ideal(self, utilities):
        m = len(utilities)
        num = 1 - self.patience ** m * (1 - self.stop) ** m
        den = m * (1 - self.patience * (1 - self.stop))
        return np.full(m, num / den)


class logarithmic:
    def _weight(self, x):
        return np.reciprocal(np.log2(np.maximum(x, 2)))

    def __call__(self, rec_df):
        rank = rec_df['rank']
        return self._weight(rank)

    def ideal(self, utilities):
        m = len(utilities)
        val = np.mean(self._weight(np.arange(m) + 1))
        return np.full(m, val)

  
class geometric:
    def __init__(self, stop=0.5):
        self.stop = stop

    def _weight(self, x):
        left = self.stop
        base = 1 - self.stop #1 - p
        exp = x - 1 #j - 1
        right = base ** exp
        return left * right
    
    def __call__(self, rec_df):
        return self._weight(rec_df['rank'])
   
    def ideal(self, utilities):
        m = len(utilities)
        val = np.mean(self._weight(np.arange(m) + 1)) 
        return np.full(m, val)


class equality:
    def __call__(self, rec_df):
        m = len(rec_df['rank'])
     
        return pd.Series(1 / m, index=rec_df.index)

    def ideal(self, utilities):
        m = len(utilities)

        return np.full(m, 1 / m)

