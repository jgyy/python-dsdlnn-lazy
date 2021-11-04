"""
Train a neural network in just 3 lines of code!
"""
from process import get_data
from sklearn.neural_network import MLPClassifier


def main():
    """
    main function
    """
    x_train, y_train, x_test, y_test = get_data()
    model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000)
    model.fit(x_train, y_train)
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
