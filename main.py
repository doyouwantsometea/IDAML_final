import argparse
from utils import preprocess
from sklearn.model_selection import train_test_split


def params():
    parser = argparse.ArgumentParser(
        description='IDA-ML Final Project'
    )

    parser.add_argument(
        '--baseline', dest='baseline',
        help='Train and evaluate baseline model.',
        action='store_true'
    )

    return parser.parse_args()


def main():
    args = params()
    if args.baseline:
        X, y = preprocess('data/fires.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(X_train.head())
        print(X_test.head())
        print(y_train.head())
        print(y_test.head())




if __name__ == '__main__':
    main()