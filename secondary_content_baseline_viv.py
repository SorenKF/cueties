from collections import defaultdict

import pandas as pd

# Define necessary filepaths.
parc_filepath = '../2020_NLP/parc_features/parc_dev_features.tsv'
polnear_filepath = '../2020_NLP/polnear_features/polnear_dev_features.tsv'


def get_dataframe(tsv_filepath):
    """
    Read in a tsv_file as a DataFrame.

    :param tsv_filepath: path to a tsv-file
    :return: df (DataFrame with the information from the tsv-file)
    """
    df = pd.read_csv(tsv_filepath, delimiter='\t', index_col=0, low_memory=False)
    return df


def get_candidate_indices(df):
    """
    Get a dictionary of all candidate indices from a tsv-file.

    :param data:
    :return: content_dictionary (a dictionary with the indices of the content spans,
                formatted as a tuple (filename, sent_no) as the key and a list of the content span indices as the value)
             content_df (a dataframe that only contains the information of the content spans)
    """
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

    # Add tuple (filename, sent_num) to the dictionary keys with index as value.
    for filename, sent_num, index in zip(content_df.filename, content_df.sentence_number, content_df.index):
        content_dictionary[filename, sent_num].append(index)

    return content_dictionary, content_df


def get_cue_from_spans(content_dictionary, content_df):
    """
    Get a dictionary of all cues by looping over the content_df with contains only the necessary data.

    :param content_dictionary: dictionary with a tuple of (filename, sentence_num) as key and a list of indices
        that refer to the content as the value.
    :param content_df: DataFrame that contains only the relevant data (sentences with no quotation marks, but
        that do contain a cue.
    :return: cues_dictionary (dictionary of cue indices, in the same format as the content_dictionary with tuples
        (filename, sentence_num) as keys and a list of cue indices as the values. Often, however, there is only one
        cue in the sentence, but in order to take into account all cases, this is presented as a list.
    """
    cues_dictionary = defaultdict(list)

    # Loop over the content_dictionary and the content_df to extract the cue labels.
    for (filename, sent_num), index in content_dictionary.items():
        for index, cue_label in zip(content_df.index, content_df.cue_label):
            if cue_label == 1:
                cues_dictionary[(filename, sent_num)].append(index)

    return cues_dictionary


def split_spans_on_cue(content_dictionary, cue_dictionary):
    """
    Split the content spans that were defined previously in content_dictionary on the cues found in cue_dictionary
        to get a list of indices that appear before the cue and after the cue. This is valuable for later checking
        which of these two list is longest.

    :param content_dictionary: nested dictionary with a tuple of (filename, sentence_num) as key and a list of content
        indices as value.
    :param cue_dictionary: nested dictionary with a tuple of (filename, sentence_num) as key and a list of cue
        indices as value.
    :return: main_dictionary (a nested dictionary with tuples of (filename, sentence_num) as key and "before" and
        "after" as nested keys that both contain a list of indices.
    """
    main_dictionary = dict()

    # Loop over content_dictionary and cue_dictionary to get the tuples and lists of indices.
    for content_tuple, content_indices in content_dictionary.items():
        for cue_tuple, cue_index in cue_dictionary.items():
            # Check if the tuples are the same for content and cue and if they are define a before and after list,
            # a main_dictionary with the tuple as the key and define cue_encountered as 0.
            if content_tuple == cue_tuple:
                main_dictionary[content_tuple] = dict()
                before = list()
                after = list()
                cue_encountered = 0

                for index in content_indices:
                    # If the index is a cue, set cue_encounterd to 1.
                    if index in cue_index:
                        cue_encountered = 1
                        continue
                    # If no cue is encountered yet, add the index to the 'before' list.
                    if cue_encountered == 0:
                        before.append(index)
                    # If a cue has been encountered previously, add the index to the 'after' list instead.
                    elif cue_encountered == 1:
                        after.append(index)

                # Add both the 'before' and 'after' indices list to their respective (filename, sentence_num) tuple
                # in the main dictionary.
                main_dictionary[content_tuple]["before"] = before
                main_dictionary[content_tuple]["after"] = after

    return main_dictionary


def create_baseline_dictionary(main_dictionary):
    """
    Create a baseline dictionary based on the main_dictionary with before and after indices. The baseline dictionary
    takes a list of indices that appear either before or after the cue, depending on which one is the longest.
    This list is meant to indicate the content span.

    :param main_dictionary: nested dictionary with a tuple of (filepath, sentence_num) as key and two lists with
        indices for 'before' and 'after' the cue.
    :return: main_dictionary (a dictionary with a tuple of (filepath, sentence_num) as key and a list with indices
        that indicate the longest span either before or after the cue, which are meant to indicate the content.
    """
    # Loop over the entries in the main_dictionary to check whether the 'before' or 'after span is longer
    # Whichever is longer is added to the dictionary as the value, with (filepath, sentence_num) as key.
    for tuple, entry in main_dictionary.items():
        if len(entry["before"]) > len(entry["after"]):
            main_dictionary[tuple] = entry["before"]
        else:
            main_dictionary[tuple] = entry["after"]
    return main_dictionary


def IOB_baseline(df, baseline_dictionary):
    """
    Adds a columns to the original DataFrame with IOBE information for the baseline content spans
        as defined in baseline_dictionary.

    :param df: original DataFrame
    :param baseline_dictionary: dictionary with (filename, sentence_num) tuples as keys
        and a list of indices as value.
    :return: df (DataFrame with IOB information for each predicted content span)
    """
    # Set default value of baseline at "O" for 'outside'
    df["baseline"] = "O"

    for tuple, indices in baseline_dictionary.items():
        df.at[indices[0], "baseline"] = "B"  # Assign B to each initial index
        df.at[indices[-1], "baseline"] = "E"  # Assign E to each final index
        df.at[indices[1:-1], "baseline"] = "I"  # Assign I to all other indices in the list

    return df


def run():
    df = get_dataframe(parc_filepath)
    content_dictionary, content_df = get_candidate_indices(df)
    cues_dictionary = get_cue_from_spans(content_dictionary, content_df)
    main_dictionary = split_spans_on_cue(content_dictionary, cues_dictionary)
    baseline_dictionary = create_baseline_dictionary(main_dictionary)
    IOB_df = IOB_baseline(df, baseline_dictionary)


if __name__ == "__main__":
    run()
