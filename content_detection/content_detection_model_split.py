# If you have not done so already: pip install sklearn-crfsuite
# To run script type in terminal:
# python train_folder test_folder out_folder
# For example:
# python content_detection_model.py ./../../Data/parc/parc_train_set/ ./../../Data/parc/parc_dev_set/ ./../../Data/parc/parc_dev_pred_set/
import sklearn_crfsuite
from csv import DictReader, DictWriter
import sys
import os
import glob


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
        token_dict.pop('attribution')
        # Append changed dict to list and label to corresponding list
        feature_dicts.append(token_dict)
        label_list.append(label)
    return feature_dicts, label_list


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
    assert len(X_test) == len(y_test) == len(y_pred), 'X_test, y_test and y_pred are not same length'
    assert len(X_test[0]) == len(y_test[0]) == len(y_pred[0]), 'X_test[0], y_test[0] and y_pred[0] are not same length'
    assert 1 == len(y_test[0][0]) == len(y_pred[0][0]), 'len of labels in y_test and y_pred are not the same'

    test_pred_data = []
    for doc_i in range(len(X_test)):
        doc_pred_data = []
        for token_i in range(len(X_test[doc_i])):
            token_dict = X_test[doc_i][token_i]

            token_dict['content_label_gold'] = y_test[doc_i][token_i]
            token_dict['content_label_pred'] = y_pred[doc_i][token_i]

            doc_pred_data.append(token_dict)
        test_pred_data.append(doc_pred_data)
    return test_pred_data


def write_list_dicts_to_folder(path_out, list_of_dicts):
    """
    Writes list of dictionaries to a folder. Make sure folder does NOT already exist.
    :param filepath_out: file to which to write the data
    :param list_of_dicts: list of dictionaries. Must all have the same keys
    """

    for doc_i in range(len(list_of_dicts)):
        filename = list_of_dicts[doc_i][0]['filename']
        filepath = path_out + filename
        with open(filepath, 'w', newline='\n', encoding='utf-8') as outfile:
            writer = DictWriter(outfile, fieldnames=list_of_dicts[doc_i][0].keys(), delimiter='\t')
            writer.writeheader()
            for data in list_of_dicts[doc_i]:
                writer.writerow(data)


def collect_feats_labels(folder_path):
    # Get training data files
    files = glob.glob(folder_path + '*')
    X_all = []
    y_all = []
    for filepath in files:
        data = read_data(filepath)
        # Separate data into features and labels
        X_doc, y_doc = extract_features_and_labels(data)
        X_all.append(X_doc)
        y_all.append(y_doc)
    return X_all, y_all


def main():
    """
        Trains a Conditional Random Field classifier on training data and predicts it on test data. The test labels are
        written to a file with the test features and true/gold labels.
        Takes the following arguments from terminal:
        - Folder path to training data
        - Folder path to test data
        - Folder path to write test data with predictions to.
        For example, (PARC)
        python content_detection_model_split.py ./../../Data/parc_split/parc/parc_train_set/ ./../../Data/parc_split/parc/parc_test_set/ ./../../Data/parc_split/parc/parc_test_pred_set/
        (PolNeAR)
        python content_detection_model_split.py ./../../Data/polnear_split/polnear/polnear_train_set/ ./../../Data/polnear_split/polnear/polnear_test_set/ ./../../Data/polnear_split/polnear/polnear_test_pred_set/
        """
    path_train = sys.argv[1]
    path_test = sys.argv[2]
    path_out = sys.argv[3]

    # Makes folder to write files to. Code is placed here because if the folder already exists then it can break here
    # rather than after having read everything in and trained the model.
    os.mkdir(path_out)
    print(f'Succesfully created directory: {path_out}')

    # Print for checks
    print(f'Training on {path_train}')
    print(f'Predicting on {path_test}')

    X_train, y_train = collect_feats_labels(path_train)
    X_test, y_test = collect_feats_labels(path_test)

    # Train the model
    crf = train_crf(X_train, y_train)
    print('Model has been fit')
    # Predict on test data
    y_pred = crf.predict(X_test)

    # Join data for writing
    true_pred_data = join_test_gold_pred(X_test, y_test, y_pred)

    # Write to folder
    write_list_dicts_to_folder(path_out, true_pred_data)
    print(f'Predicted labels written to {path_out}')

if __name__ == '__main__':
    main()
