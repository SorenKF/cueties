from collections import defaultdict
import pandas as pd
import os

def collect_candidate_cues(training_corpus):
    '''
    Takes the path to a training corpus and returns a dictionary of the cue tokens together
    with a list [occurrences_as_cue, total_occurrences]
    
    :param training_corpus: the path to a directory containing the training files
        training files should be .tsv files of pandas dfs with at least columns "token" and "cue_label"
    
    :returns cue_dict: a dictionary of cue_token:[occurrences_as_cue, total_occurrences]
    '''
    token_dict = defaultdict(lambda: [0,0]) # Collect counts for all tokens in corpus
    
    for file in os.listdir(training_corpus):
        df = df = pd.read_csv(training_corpus+file, sep='\t', header=0)
        tokens = df["token"]
        cue_labels = df["cue_label"]
        for token, cue_label in zip(tokens, cue_labels):
            token_dict[token][0] += cue_label
            token_dict[token][1] += 1
    
    cue_dict = dict() # Collect only cues from token_dict
    for token, counts in token_dict.items():
        if counts[0] > 0:
            cue_dict[token] = counts
    return cue_dict
    
def add_lexicon_check(df, lexicon):
    '''
    Takes a pandas df with at least a "token" column and a lexicon (a set of tokens) and adds a column 
    to the df indicating for each token in the df whether it is in the lexicon or not
    
    :param df: a pandas dataframe with at least a "token" column
    :param lexicon: a set of words
    '''
    in_lexicon = []
    for token in df["token"]:
        if token in lexicon:
            in_lexicon.append(1)
        else:
            in_lexicon.append(0)
    df["lexicon_check"] = in_lexicon
    
    