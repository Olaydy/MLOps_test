import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--test_size', type=float)
    args = arg_parser.parse_args()
    
    dataset = pd.read_csv('../data/interim/features_iris.csv')
    
    # Split in train/test

    df_train, df_test = train_test_split(dataset, test_size=args.test_size, random_state=42)
    
    df_train.to_csv('../data/external/train.csv', index=False)
    df_test.to_csv('../data/external/test.csv', index=False)
    
