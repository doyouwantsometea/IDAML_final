import argparse
from sklearn.model_selection import train_test_split
from model import baseline_model, single_model, classification
from utils import preprocess


def params():
    parser = argparse.ArgumentParser(
        description='IDA-ML Final Project'
    )

    parser.add_argument(
        '--baseline', dest='baseline',
        help='Train and evaluate baseline model.',
        action='store_true'
    )

    parser.add_argument(
        '--single_model', dest='single_model',
        help='Train and evaluate the simple model.',
        action='store_true'
    )

    parser.add_argument(
        '--multi_layer', dest='multi_layer',
        help='Train and evaluate the model with multi-layer architecture.',
        action='store_true'
    )

    return parser.parse_args()


def main():
    args = params()
    if args.baseline:
        X, y = preprocess('data/fires.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # print(X_train.head())
        # print(X_test.head())
        # print(y_train.head())
        # print(y_test.head())
        baseline_model(X_train, X_test, y_train, y_test)

    if args.single_model:
        X, y = preprocess('data/fires.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # print(X_train.head())
        # print(X_test.head())
        # print(y_train.head())
        # print(y_test.head())
        single_model(X_train, X_test, y_train, y_test)

    if args.multi_layer:
        X, y = preprocess('data/fires.csv', add_class=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = classification(X_train, X_test, y_train, y_test)
        single_model(X_train, X_test, y_train, y_test, add_layer=True)
        



if __name__ == '__main__':
    main()