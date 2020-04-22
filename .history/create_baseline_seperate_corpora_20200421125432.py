import pandas as pd
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)


def get_cue_list(train_folder):
    """
    Get a list of all the cues used in the training folder.

    :parameter: train_folder (path to the folder that contains the training information)
    :return: cue list (list of cues used in the folder)
    """
    # Loop over the files in the preprocessed folder to extract the lemma of the cues.
    for filename in os.listdir(train_folder):
        data = pd.read_csv(train_folder + filename, delimiter="\t")
        data = data[["lemma", "cue_label"]]
        cue_data = data[data["cue_label"] == 1]
        cue_list = cue_data["lemma"].tolist()
    return cue_list


def get_common_cues(parc_train_folder, polnear_train_folder, k):
    """
    Create a list of the k most common cues, based on the full list gathered from the pre-processed training data.

    :parameter: parc_train_folder (folder to the training folder for parc)
    :parameter: polnear_train_folder (folder to the training folder for polnear)
    :parameter: k (number of cues that should be returned)
    :returns common_cue_list (list of the k most common cues.
    """
    cue_freq_dict = dict()

    parc_cue_list = get_cue_list(parc_train_folder)
    polnear_cue_list = get_cue_list(polnear_train_folder)

    # Create a frequency dictionary to count the number of occurrences for each cue.
    for cue in parc_cue_list:
        if cue in cue_freq_dict.keys():
            cue_freq_dict[cue] += 1
        elif cue not in cue_freq_dict.keys():
            cue_freq_dict[cue] = 1

    for cue in polnear_cue_list:
        if cue in cue_freq_dict.keys():
            cue_freq_dict[cue] += 1
        elif cue not in cue_freq_dict.keys():
            cue_freq_dict[cue] = 1

    # Sort the frequency dictionary from the highest to the lowest number of occurrences.
    sorted_cue_counter = sorted(
        cue_freq_dict.items(), key=lambda item: item[1], reverse=True)

    # Only select the k most frequent cues and write them to a list.
    common_cue_list = list()
    for (lemma, count) in sorted_cue_counter[:k]:
        common_cue_list.append(lemma)

    logging.info(f"These are the {k} most common cues: {common_cue_list}")

    return common_cue_list


def add_baseline_column(test_folder, cues):
    """
    Adds a column to the original test data with baseline information.
        If the lemma appears in the k most common cue list (cues), the system will ascribe a 1 (True);
        if not the system will give a 0 (False).

    :parameter: test_folder (path to the folder that contains the test information)
    :parameter: cues (list of the most common cues, extracted from the parc and polnear training data)
    :return: test_data (pd.DataFrame that contains the test data (lemma, cue_label) with added baseline information.
    """
    test_data = pd.DataFrame()

    # Loop over the files in the test folder to read in information on lemma and cue_label.
    for filename in os.listdir(test_folder):
        data = pd.read_csv(test_folder + filename, delimiter="\t")
        data = data[["lemma", "cue_label"]]

        # Get baseline information and transform it to a binary format.
        data["baseline"] = data["lemma"].isin(cues)
        translation = {True: 1, False: 0}
        data.baseline = data.baseline.map(translation)

        # Add information to a larger dataframe.
        test_data = test_data.append(data, ignore_index=True)

    return test_data


def add_baseline_info(corpus_test_folder, cues, baseline_path):
    """
    Combines the information gathered in add_baseline_column for both parc and polnear to a single dataframe,
        to prepare it for evaluation, and writes it to a csv file.

    :parameter: corpus_test_folder (path to a corpus test folder)
    :parameter: baseline_path (path to the new file to which the baseline will be written)
    :parameter: k (specifies the number of cues in the common cue list)
    """
    # Add baseline column
    corpus_test_folder = add_baseline_column(corpus, cues)

    # Add individual dataframes together and write to file.
    complete_df = parc_test_data.append(corpus_test_folder, ignore_index=True)
    complete_df.to_csv(baseline_path, index=False, index_label=False)


def evaluate_baseline(baseline_data_path):
    """
    Evaluation metrics for the baseline (accuracy, precision and recall).
    Check whether the lemma in question matches with any of the previously extracted cues in the most common cue list

    :parameter: baseline_data_path (path to the csv-file that contains the baseline information).
    """
    # Read in the baseline data.
    data = pd.read_csv(baseline_data_path)

    # Read in the gold labels (cue_label) and the predicted labels (baseline).
    y_true = data["cue_label"].to_numpy()
    y_pred = data["baseline"].to_numpy()

    # Get evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    logging.info(
        f"Baseline accuracy score: {round(accuracy, 3)} "
        f"\nBaseline precision score: {round(precision, 3)} "
        f"\nBaseline recall score: {round(recall, 3)}")


def run():
    parc_train_folder = "../Data/updated_corpora/parc_train_updated/"
    polnear_train_folder = "../Data/updated_corpora/polnear_train_updated/"

    parc_test_folder = "../Data/updated_corpora/parc_test_updated/"
    polnear_test_folder = "../Data/updated_corpora/polnear_test_updated/"
    parc_baseline_data_path = "./baseline/parc_baseline.csv"
    polnear_baseline_data_path = './baseline/polnear_baseline.csv'

    cues = get_common_cues(parc_train_folder, polnear_train_folder, 10)
    add_baseline_info(parc_test_folder,
                      cues, parc_baseline_data_path)
    add_baseline_info(polnear_test_folder, cues, polnear_baseline_data_path)

    evaluate_baseline(parc_baseline_data_path)
    evaluate_baseline(polnear_baseline_data_path)

if __name__ == "__main__":
    run()
