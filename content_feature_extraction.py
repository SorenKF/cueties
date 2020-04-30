# Main file for feature extraction for content detection
# Takes a .tsv containing a pandas df and prints a new file with feature columns added
# To run: python content_feature_extraction input_filepath output_filepath


import pandas as pd
import sys

def extract_content_labels(df):
    '''
    Takes an attribution pandas df and returns a list of gold content labels
    
    :param attribution_df: a pandas dataframe with conll attribution column format, with a column "attribution"
    
    :returns content_label_gold: a list of gold content labels
    '''
    content_labels = []
    for index, attribution_info in enumerate(df["attribution"]):
        if attribution_info == "0": # This is the case when there was no attribution in the original file
            content_labels.append("O")
            continue
        labelled = False
        for attribution in attribution_info.split(" "):
            attribution_split = attribution.split("-")
            if attribution != "_" and attribution_split[2] != "NE" and attribution_split[1] == "CONTENT":
                attribution_label = attribution[0] # I/B label
                if not labelled:
                    content_labels.append(attribution_label)
                    labelled = True
        if not labelled:
            content_labels.append("O")
            
    return content_labels
            
                
def ancestor_is_cue(df):
    '''
    Takes an attribution df and returns a list of binary labels for "dependency ancestor is cue"
    
    :param attribution_df: a pandas dataframe with conll attribution column format
    
    :returns ancestor_is_cue_labels: a list of labels
    '''
    ancestor_is_cue_labels = []
    current_index_in_sentence = 0
    sentence = dict() # stores current sentence, in the form {number_in_sent: (dep_head, cue_label), ...})
    
    for number_in_sent, cue_label, dep_head in zip(df["sentence_token_number"], df["cue_label"], df["dependency_head"]):
        if int(number_in_sent) < current_index_in_sentence:
            labels = ancestor_labels(sentence)
            #DEBUG
            assert labels != None, "Error in dependency structure of sentence. Aborting."
            ancestor_is_cue_labels += labels
            sentence = dict()
        current_index_in_sentence = number_in_sent
        sentence[number_in_sent] = (dep_head, cue_label)
    #For last sentence:
    labels = ancestor_labels(sentence)
    ancestor_is_cue_labels += labels
    return ancestor_is_cue_labels

def ancestor_labels(sentence):
    '''
    Takes a sentence and returns a list of labels representing for each token in the sentence 
    whether any of its ancestors is a cue
    
    :param sentence: a dictionary of the form {number_in_sent: (dep_head, cue_label), ...}
    
    :returns labels: a list of binary labels
    '''
    labels = [0] * len(sentence)
    
    for index, t in sentence.items():
        dep_head, cue_label = t
        
        i = 0 #DEBUG
        while True:
            #DEBUG
            i+=1
            if i > 100:
                return None
            #ENDDEBUG
            if cue_label == 1:
                labels[index-1]=1
                break
            if dep_head == 0:
                break
            else:
                dep_head, cue_label = sentence[dep_head]
    assert len(labels)==len(sentence)
    return labels
            
def cue_in_sentence(df):
    '''
    Takes an attribution df and returns a list of binary labels for "cue in sentence"
    
    :param attribution_df: a pandas dataframe with conll attribution column format
    
    :returns cue_in_sentence_labels: a list of labels
    '''
    
    cue_in_sentence_labels = []
    current_index_in_sentence = 0
    sentence = dict() # stores current sentence, in the form {number_in_sent: (dep_head, cue_label), ...})
    
    for number_in_sent, cue_label in zip(df["sentence_token_number"], df["cue_label"]):
        if int(number_in_sent) < current_index_in_sentence:
            label = 1 if (1 in sentence.values()) else 0
            cue_in_sentence_labels += [label] * len(sentence)
            sentence = dict()
        current_index_in_sentence = number_in_sent
        sentence[number_in_sent] = cue_label
    #For last sentence:
    label = 1 if (1 in sentence.values()) else 0
    cue_in_sentence_labels += [label] * len(sentence)
    return cue_in_sentence_labels

def main():
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    df = pd.read_csv(input_path, sep='\t', header=0, index_col=0)
    
    content_labels = extract_content_labels(df)
    df["content_label_gold"] = content_labels
    
    ancestor_is_cue_labels = ancestor_is_cue(df)
    df["ancestor_is_cue"] = ancestor_is_cue_labels
    
    cue_in_sentence_labels = cue_in_sentence(df)
    df["cue_in_sentence"] = cue_in_sentence_labels
    
    # Stella's function calls here
    
    df.to_csv(output_path, sep="\t")
    