'''
./src/models/*.py
./data/*.csv

'''


import os
import pandas as pd

import argparse

from src.models import SVD


### log

def main():
    
    
    ######################## DATA LOAD
    
    
    if arg.ENSEMBLE_STRATEGY:
        for data_path in arg.ENSEMBLE_DATA_PATH:
            a = pd.read_csv(data_path)
        
    else:
        for data_path in arg.ENSEMBLE_DATA_PATH:
            a = pd.read_csv(data_path)
        
    
    
    
    ######################## Train/Valid Split
    
    
    
    
    
    ######################## Model
    
    
    
    
    
    ######################## TRAIN
    
    
    
    
    
    ######################## INFERENCE
    
    
    
    
    
    ######################## SAVE
    
    
    


if __name__ == "__main__":
    
    
    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    
    arg('--DATA_PATH', type=str, default='data/')
    arg('', type=str, choices=['SVD', 'MF', 'ALS', 'DeepConv'])
    
    
    
    
    ######################## ENSEMBLE ENVIRONMENT SETUP
    arg('--ENSEMBLE_DATA_PATH', type=list, help="delimited list input")
    arg('--ENSEMBLE_MODEL_PATH', type=list, help="delimited list input")
    arg('--ENSEMBLE_STRATEGY', type=str, default='weighted_avg')
    
    
    args = parser.parse_args()
    
    
    main(args)