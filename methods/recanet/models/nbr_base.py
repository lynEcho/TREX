import sys

class NBRBase:
    def __init__(self, train_baskets, test_baskets, valid_baskets):
        self.train_baskets = train_baskets
        self.test_baskets = test_baskets
        self.valid_baskets = valid_baskets
        
        '''
        basket_per_user = self.train_baskets[['user_id','basket_id']].drop_duplicates() \
            .groupby('user_id').agg({'basket_id':'count'}).reset_index()
    
        self.test_users = basket_per_user[basket_per_user['basket_id'] >= self.basket_count_min]['user_id'].tolist() #
        print(self.basket_count_min)
        print("number of test users:", len(self.test_users))
        
        item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name = 'item_count').reset_index()
        item_counts = item_counts[item_counts['item_count']>= min_item_count] #didn't filter 
        print(min_item_count)

        self.item_counts_dict = dict(zip(item_counts['item_id'],item_counts['item_count']))
        print("filtered items:", len(self.item_counts_dict))
        '''
        #new
        self.test_users = self.train_baskets['user_id'].drop_duplicates().tolist() #all users

        item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name = 'item_count').reset_index()
        self.item_counts_dict = dict(zip(item_counts['item_id'],item_counts['item_count']))
        
    def train(self):
        pass

    def predict(self):
        pass
