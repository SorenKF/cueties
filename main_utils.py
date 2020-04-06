import pandas as pd
import os


def import_attribution_doc(filename):
    '''
    Takes the name of a PARC or polnear attribution corpus document and returns a pandas df containing its data
    
    :param filename: the path to a file; .tsv conll format
    
    :returns df: a pandas dataframe with columns corresponding to those in the file:
            "filename", "sentence_number", "doc_token_number", "sent_token_number", "offset", "word", "lemma", "POS", "dependency_label",
                "dependency_head_id", "attribution"
    '''

    headers = ["filename", "sentence_number", "doc_token_number", "sent_token_number", "offset",
               "word", "lemma", "POS", "dependency_label", "dependency_head_id", "attribution"]
    df = pd.read_csv(filename, sep='\t', names=headers)
    return df


def extract_attributions(attribution_df):
    '''
    Takes a pandas df containing information from an attribution corpus file and returns
    a list of attributions in the form of dictionaries of span tuples.
    
    :param attribution_df: a pandas df with info from an attribution file
    
    :returns attributions: a list of dictionaries of lists of span tuples. An attribution consists of a source,
        a cue, and a content, and each of these has one or more associated span (start and end indices). The
        span indices represent the location of the source, cue or content in the df. "s" is the first
        df row of the span, "e" is the line in the df after the span ends (use range(s, e) or df[s:e] to traverse the span)
        attributions = [{"SOURCE": [(s, e)], "CUE": [(s, e)], "CONTENT": [(s1, e1), (s2, e2)]}, ...]
    '''
    filename = attribution_df["filename"][0]

    attribution_info = attribution_df["attribution"]
    num_attributions = len(attribution_info[0].split(" "))
    attributions = []
    attribution_parts = ["SOURCE", "CUE", "CONTENT"]
    for i in range(num_attributions):
        attributions.append({"SOURCE": [], "CUE": [], "CONTENT": []})

    for att_index in range(num_attributions):
        for attribution_part in attribution_parts:
            start_index = None
            for i in range(len(attribution_info)):
                attribution_string = attribution_info[i].split(" ")[att_index].split("-")
                if attribution_string == ["_"] or attribution_string[1] != attribution_part:
                    # Not inside a span
                    if start_index != None:
                        # Found the end of a span we were in
                        attributions[att_index][att_part].append((start_index, i))
                        start_index = None
                else:
                    IOB_label, att_part = attribution_string[0], attribution_string[1]
                    if att_part == attribution_part and IOB_label == "B":
                        start_index = i

    return attributions


# Get span of attributions
def extract_attribution_spans(attributions):
    """
    Extracts the lowest and highest index from an attribution to find the span of the attribution.
    Returns a list of tuples.

    :param attributions: list of dictionaries with values being lists of tuples
    """
    attribution_spans = []
    for attribution in attributions:
        lowest_value = 9999999
        highest_value = 0
        for tuple_list in attribution.values():
            for attribution_tuple in tuple_list:
                if type(attribution_tuple) == tuple and len(attribution_tuple) == 2:  # To prevent code breaking
                    start_index, end_index = attribution_tuple
                    if start_index < lowest_value:
                        lowest_value = start_index
                    if end_index > highest_value:
                        highest_value = end_index
            attribution_spans.append((lowest_value, highest_value))

    return attribution_spans


def import_attribution_tsv(attribution_tsv):
    '''
    Imports an attribution tsv file into a pandas dataframe. 
    Columns: filename, SOURCE, CUE, CONTENT
    
    :param attribution_tsv: a .tsv file containing attribution information (in span-tuple-list format) for a corpus of files
    ex.
    filename    SOURCE    CUE    CONTENT
    wsj_0001.txt    1,3    4,7   19,29;35,39     
    ...
    
    :returns df: a pandas dataframe with columns: filename, SOURCE, CUE, CONTENT
        SOURCE, CUE, CONTENT are represented as span (start, end) tuples.
    '''
    df = pd.read_csv(attribution_tsv, sep='\t')
    return df
