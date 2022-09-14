'''
./src/models/*.py
./data/*.csv

'''


import os
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix


import argparse

from src import SVD

from src import seed_everything


### log

def main(args):
    seed_everything(args.SEED)
    
    
    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + '/users.csv')
    books = pd.read_csv(args.DATA_PATH + '/books.csv')
    train = pd.read_csv(args.DATA_PATH + '/train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + '/test_ratings.csv')
    
    print(users)
    print(books)
    print(train)
    print(test)

    if args.ENSEMBLE_STRATEGY:
        for data_path in args.ENSEMBLE_DATA_PATH:
            df = pd.read_csv(data_path)
        
    ######################## PREPROCESS
    
    ######################## USER, BOOK Key value
    ids = pd.concat([train['user_id'], test['user_id']]).unique()
    isbns = pd.concat([train['isbn'], test['isbn']]).unique()
    
    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}
    
    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}
    
    train['user_id'] = train['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    
    train['isbn'] = train['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    
    
    ############### test를 위해서 임시 test 생성
    df_test = test.copy()
    df_test['rating'] = 0
    
    train = pd.concat([train, df_test], axis=0).reset_index(drop=True)
    print(train)
    
    # Conversion via COO matrix
    train_csr = csr_matrix((train['rating'], (train['user_id'], train['isbn'])), shape=(len(user2idx), len(isbn2idx)))
    
    ######################## Train/Valid Split
    if args.MODEL=='SVD':
        print(type(train_csr))
        print(train_csr.shape)
        
    
    elif args.MODEL=='DeepConv':
        pass
    
    
    
    ######################## Model
    if args.MODEL=='SVD':
        model = SVD(train, truncate=2)
    
    ######################## TRAIN
    model.train()
    
    ######################## INFERENCE
    if args.MODEL=='SVD':
        restore_df = model.predict()
    
    
    ######################## SAVE
    if args.MODEL=='SVD':
        df_train = pd.read_csv(args.DATA_PATH + '/train_ratings.csv')
        test['pred_rating'] = restore_df[len(df_train):, 2]
        test.to_csv('submit/SVD.csv', index=False)
    
    
    


if __name__ == "__main__":
    
    
    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    
    arg('--DATA_PATH', type=str, default='data/')
    arg('--MODEL', type=str, choices=['SVD', 'MF', 'ALS', 'DeepConv'])
    
    
    arg('--SEED', type=int, default=42)
    
    
    
    
    ######################## ENSEMBLE ENVIRONMENT SETUP
    arg('--ENSEMBLE_DATA_PATH', type=list, default=[], help="delimited list input")
    arg('--ENSEMBLE_MODEL_PATH', type=list, default=[], help="delimited list input")
    arg('--ENSEMBLE_STRATEGY', type=str, default='weighted_avg')
    
    
    args = parser.parse_args()
    main(args)