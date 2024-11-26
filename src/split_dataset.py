import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


def split(input_csv: str, test_size: float, train_csv: str, test_csv: str):
    """Split data to train and test
    Parameters
    ----------
    input_csv : `str`
        File csv for split
    test_size : `float`
        Size of test dataset [0,1]
    train_csv : `int`
        Output file csv for train dataset
    test_csv : `int`
        Output file csv for trest dataset
    """
    dataset = pd.read_csv(input_csv)

    # Split in train/test
    df_train, df_test = train_test_split(dataset, test_size=test_size, random_state=42)

    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--test_size", type=float, default=0.3)
    arg_parser.add_argument(
        "--input_csv", 
        type=str, default="data/interim/features_iris.csv"
        )
    arg_parser.add_argument("--train_csv",
        type=str,
        default="data/external/train.csv"
        )
    arg_parser.add_argument("--test_csv", 
        type=str,
        default="data/external/test.csv"
        )
    args = arg_parser.parse_args()

    split(args.input_csv, args.test_size, args.train_csv, args.test_csv)