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
    df = pd.read_csv(filename, sep='\t',names=headers)
    return df

