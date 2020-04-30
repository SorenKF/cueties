# If you have not done so already: pip install sklearn-crfsuite
# To run script type in terminal:
# python train_file test_file out_file
# For example:
# python content_detection_model.py ./../../Data/parc_features/parc_train_features.tsv ./../../Data/parc_features/parc_test_features.tsv ./../../Data/parc_features/parc_test_predicted_features.tsv
import sklearn_crfsuite
from csv import DictReader, DictWriter
import sys


def extract_features_and_labels(list_of_dicts):
    """
    Separates list of dictionaries to list of dictionaries with features and list of lables. It is a nested list
    because that is required for the CRF.

    :param list_of_dicts: list of dictionaries. All dictionaries must have the same keys. All dicts must have
    'content_label_gold' key
    :return: list of list of dictionaries containing features and list of list of strings containing labels
    """
    # Define objects
    feature_dicts = []
    label_list = []
    # Loop through dictionaries in data set
    for token_dict in list_of_dicts:
        # Remove label from dictionaries
        label = token_dict.pop('content_label_gold')
        # Append changed dict to list and label to corresponding list
        feature_dicts.append(token_dict)
        label_list.append(label)
    return [feature_dicts], [label_list]


def train_crf(X_train, y_train):
    """
    Defines and fits Conditional Random Field algorithm. Sets algorithm to 'lbfgs'.
    :param X_train: List of list of dictionaries containing features
    :param y_train: List of list of strings/labels
    :return: fitted model
    """
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf = crf.fit(X_train, y_train)
    return crf


def read_data(filepath):
    """
    Reads tsv file and returns list of dicts with header of tsv as keys.
    :param filepath: str, filepath to read
    :return: list of dicts
    """
    with open(filepath, 'r', encoding='utf-8') as infile:
        dict_reader = DictReader(infile, delimiter='\t')
        data = list(dict_reader)
    return data


def join_test_gold_pred(X_test, y_test, y_pred):
    """
    Joins test features, true labels and predicted labels to one big list of dicts.
    :param X_test: List of list of dicts containing the features of the test set
    :param y_test: list of list of str which are the true labels of the test set
    :param y_pred: list of list of str which are the predicted labels of the test set
    :return: list of dictionaries containing test features, true labels and predicted labels
    """
    assert len(X_test[0]) == len(y_test[0]), 'X_test[0] and y_test[0] are not same length'
    assert len(y_test[0]) == len(y_pred[0]), 'y_test[0] and y_pred[0] are not same length'
    test_pred_data = []
    for i in range(len(X_test[0])):
        token_dict = X_test[0][i]

        token_dict['content_label_gold'] = y_test[0][i]
        token_dict['content_label_pred'] = y_pred[0][i]
        test_pred_data.append(token_dict)
    return test_pred_data


def write_dicts_to_file(filepath_out, list_of_dicts):
    """
    Writes list of dictionaries to a file
    :param filepath_out: file to which to write the data
    :param list_of_dicts: list of dictionaries. Must all have the same keys
    """
    with open(filepath_out, 'w', newline='\n', encoding='utf-8') as outfile:
        writer = DictWriter(outfile, fieldnames=list_of_dicts[0].keys(), delimiter='\t')
        writer.writeheader()
        for data in list_of_dicts:
            writer.writerow(data)


def main():
    """
    Trains a Conditional Random Field classifier on training data and predicts it on test data. The test labels are
    written to a file with the test features and true/gold labels.
    Takes the following arguments from terminal:
    - Filename to training data
    - Filename to test data
    - Filename to write test data with predictions to.
    For example, (PARC)
    python content_detection_model.py ./../../Data/parc_features/parc_train_features.tsv ./../../Data/parc_features/parc_test_features.tsv ./../../Data/parc_features/parc_test_predicted_features.tsv
    (PolNeAR)
    python content_detection_model.py ./../../Data/polnear_features/polnear_train_features.tsv ./../../Data/polnear_features/polnear_test_features.tsv ./../../Data/polnear_features/polnear_test_predicted_features.tsv
    """
    filename_train = sys.argv[1]
    filename_test = sys.argv[2]
    filepath_out = sys.argv[3]

    # Print for checks
    print(f'Training on {filename_train}')
    print(f'Predicting on {filename_test}')

    # Read in data
    train_data = read_data(filename_train)
    test_data = read_data(filename_test)

    print('Read data. Proceed to extract features and labels')
    # Separate data into features and labels
    X_train, y_train = extract_features_and_labels(train_data)
    X_test, y_test = extract_features_and_labels(test_data)

    print('Extracted features and labels. Proceed to train model.')

    # Train the model
    crf = train_crf(X_train, y_train)
    print('Model has been fit')
    # Predict on test data
    y_pred = crf.predict(X_test)

    print('Labels have been predicted on test. Continue to write output to file.')
    # Join features, gold labels and predicted labels
    true_pred_data = join_test_gold_pred(X_test, y_test, y_pred)
    # Write to file
    write_dicts_to_file(filepath_out, true_pred_data)

    print(f'Predicted labels written to {filepath_out}')


if __name__ == '__main__':
    main()
