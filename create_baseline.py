import pandas as pd
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)


def get_common_cues(folder, k):
    """
    Create a list of the k most common cues found in the pre-processed data.
    :returns common_cue_list (list of the k most common cues.
    """
    cue_freq_dict = dict()

    # Loop over the files in the preprocessed folder to extract the lemma of the cues.
    for filename in os.listdir(folder):
        data = pd.read_csv(folder + filename, delimiter="\t")
        data = data[["lemma", "cue_label"]]
        cue_data = data[data["cue_label"] == 1]
        cue_list = cue_data["lemma"].tolist()
        # Create a frequency dictionary to count the number of occurrences for each cue.
        for cue in cue_list:
            if cue in cue_freq_dict.keys():
                cue_freq_dict[cue] += 1
            elif cue not in cue_freq_dict.keys():
                cue_freq_dict[cue] = 1

    # Sort the frequency dictionary from the highest to the lowest number of occurrences.
    sorted_cue_counter = sorted(cue_freq_dict.items(), key=lambda item: item[1], reverse=True)

    # Only select the k most frequent cues and write them to a list.
    common_cue_list = list()
    for (lemma, count) in sorted_cue_counter[:k]:
        common_cue_list.append(lemma)

    logging.info(f"These are the {k} most common cues: {common_cue_list}")

    return common_cue_list


def add_baseline_info(folder, baseline_path):
    cues = get_common_cues(folder, 10)

    complete_df = pd.DataFrame()

    for filename in os.listdir(folder):
        data = pd.read_csv(folder + filename, delimiter="\t")
        data = data[["lemma", "cue_label"]]
        data["baseline"] = data["lemma"].isin(cues)
        translation = {True: 1, False: 0}
        data.baseline = data.baseline.map(translation)

        complete_df = complete_df.append(data, ignore_index=True)

    complete_df.to_csv(baseline_path, index=False, index_label=False)


def evaluate_baseline(baseline_data_path):
    data = pd.read_csv(baseline_data_path)

    y_true = data["cue_label"].to_numpy()
    y_pred = data["baseline"].to_numpy()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    logging.info(
        f"Baseline accuracy score: {round(accuracy, 3)} "
        f"\nBaseline precision score: {round(precision, 3)} "
        f"\nBaseline recall score: {round(recall, 3)}")


folder = "../parc30-conll/parc_train_updated/"
baseline_data_path = "../baseline-information/baseline.csv"
add_baseline_info(folder, baseline_data_path)

evaluate_baseline(baseline_data_path)
