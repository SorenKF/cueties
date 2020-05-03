import pandas as pd
import numpy as np
from collections import defaultdict

parc_filepath = '../2020_NLP/parc_features/parc_dev_features.tsv'
polnear_filepath = '../2020_NLP/polnear_features/polnear_dev_features.tsv'


def get_df(tsv_filepath):
    df = pd.read_csv(tsv_filepath, delimiter='\t', index_col=0, low_memory=False)
    return df


def get_candidate_indices(df):
    candidate_indices = list()

    # List of tuples with whether (1) there is a quotation mark in the sentence and
    # (2) whether there is a cue in the sentence.
    factors = list(zip(df['qm_in_sent'], df['cue_in_sentence']))
    factors = enumerate(factors)

    # Loop over all entries and check whether (1) the sentence contains a quotation mark or not and
    # (2) whether there is a cue in the sentence or not. If the sentence doesn't contain a quotation mark
    # but does contain a cue, it is added to the candidate_indices list.
    for index, (qm_sent, cue_sent) in factors:
        if qm_sent != 1 and cue_sent == 1:
            candidate_indices.append(index)

    content_df = df.loc[candidate_indices]
    content_dictionary = defaultdict(list)

    for filename, sent_num, index in zip(content_df.filename, content_df.sentence_number, content_df.index):
        content_dictionary[filename, sent_num].append(index)

    return content_dictionary, content_df


def get_cue_from_spans(content_dictionary, content_df):
    cues_dictionary = defaultdict(list)

    for (filename, sent_num), index in content_dictionary.items():
        for index, cue_label in zip(content_df.index, content_df.cue_label):
            if cue_label == 1:
                cues_dictionary[(filename, sent_num)].append(index)

    return cues_dictionary


def split_spans_on_cue(content_dictionary, cue_dictionary):
    main_dictionary = dict()

    for content_tuple, content_indices in content_dictionary.items():
        for cue_tuple, cue_index in cue_dictionary.items():
            if content_tuple == cue_tuple:
                main_dictionary[content_tuple] = dict()
                before = list()
                after = list()
                cue_encountered = 0

                for index in content_indices:
                    if index in cue_index:
                        cue_encountered = 1
                        continue
                    if cue_encountered == 0:
                        before.append(index)
                    elif cue_encountered == 1:
                        after.append(index)
                main_dictionary[content_tuple]["before"] = before
                main_dictionary[content_tuple]["after"] = after

    return main_dictionary


def run():
    df = get_df(parc_filepath)
    content_dictionary, content_df = get_candidate_indices(df)
    cues_dictionary = get_cue_from_spans(content_dictionary, content_df)
    main_dictionary = split_spans_on_cue(content_dictionary, cues_dictionary)


if __name__ == "__main__":
    run()
