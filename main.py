import argparse
from sklearn.model_selection import train_test_split
from model import baseline_model, kernel_ridge, classification, evaluate
from utils import preprocess


def params():
    parser = argparse.ArgumentParser(
        description='IDA-ML Final Project'
    )

    parser.add_argument(
        '--baseline', dest='baseline',
        help='Train and evaluate baseline ridge regression model.',
        action='store_true'
    )

    parser.add_argument(
        '--kernel_ridge', dest='kernel_ridge',
        help='Train and evaluate the ridge regression model with kernel trick.',
        action='store_true'
    )

    parser.add_argument(
        '--multi_steps', dest='multi_steps',
        help='Train and evaluate the kernel ridge model with an additional logistic regression step.',
        action='store_true'
    )

    return parser.parse_args()


def main():
    args = params()
    if args.baseline:
        X, y = preprocess('data/fires.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pred = baseline_model(X_train, X_test, y_train, y_test)
        evaluate(pred, y_test)

    if args.kernel_ridge:
        X, y = preprocess('data/fires.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pred = kernel_ridge(X_train, X_test, y_train, y_test)
        evaluate(pred, y_test)

    if args.multi_steps:
        X, y = preprocess('data/fires.csv', add_class=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, y_train = classification(X_train, X_test, y_train, y_test)
        pred = kernel_ridge(X_train, X_test, y_train, y_test, add_layer=True)
        evaluate(pred, y_test)


if __name__ == '__main__':
    main()