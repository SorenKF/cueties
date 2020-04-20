import pandas as pd
from collections import Counter
import sklearn.metrics

filepath_parc = 'parc_system_output.tsv'
filepath_polnear = 'polnear_system_output.tsv'

def get_general_performance(filepath, corpus):
    """
    Function that returns precision, recall, f1 and accuracy of the system of parc or polnear
    :param filepath: a filepath to the system output    str
    :param corpus: the name of the corpus you want to get the stats for   str
    """
    
    df = pd.read_csv(filepath, sep="\t", index_col=0, header=0)

    precision = sklearn.metrics.precision_score(df["cue_label"], df["predicted_cue_label"])
    recall = sklearn.metrics.recall_score(df["cue_label"], df["predicted_cue_label"])
    f1_score = sklearn.metrics.f1_score(df["cue_label"], df["predicted_cue_label"])
    accuracy = sklearn.metrics.accuracy_score(df["cue_label"], df["predicted_cue_label"], normalize=True, sample_weight=None)
    
    print('precision ' + corpus + ': ' +str(precision)) 
    print('recall ' + corpus + ': ' + str(recall))
    print('f1_score ' + corpus + ': ' + str(f1_score))
    print('accuracy ' + corpus + ': ' + str(accuracy))

def get_fp_dicts(filepath):
    """ 
    creates dictionaries to count how often aspects of false positives occur
    :param filepath: a string
    """
        
    # creating dataframe
    df = pd.read_csv(filepath, sep='\t', header=0)
    
    # creating lists for false positives
    pos_false_positives = []
    lemma_false_positives = []
    dependency_labels_false_positives = []
    
    # looking for aspects of the cues that are false positives
    for index, prediction in enumerate(df['predicted_cue_label']):
        if df.iloc[index]['cue_label'] == 0 and df.iloc[index]['predicted_cue_label'] == 1:
            pos_false_positives.append(df['POS'].iloc[index])
            lemma_false_positives.append(df['lemma'].iloc[index])
            dependency_labels_false_positives.append(df['dependency_label'].iloc[index])
            
    # creating dictionaries to count the frequencies
    pos_fp_dict = Counter(pos_false_positives)
    lemma_fp_dict = Counter(lemma_false_positives)
    dependency_fp_dict = Counter(dependency_labels_false_positives)
    
    return(pos_fp_dict, lemma_fp_dict, dependency_fp_dict)
    
    
def get_fn_dicts(filepath):   
    """ 
    creates dictionaries to count how often aspects of false negatives occur
    :param filepath: a string
    """
    
    df = pd.read_csv(filepath, sep='\t', header=0)
    
    # creating lists for false negatives
    pos_false_negatives = []
    lemma_false_negatives = []
    dependency_labels_false_negatives = []
    
    # looking for aspects of the cues that are false negatives
    for index, prediction in enumerate(df['predicted_cue_label']):
        if df.iloc[index]['cue_label'] == 1 and df.iloc[index]['predicted_cue_label'] == 0:
            pos_false_negatives.append(df['POS'].iloc[index])
            lemma_false_negatives.append(df['lemma'].iloc[index])
            dependency_labels_false_negatives.append(df['dependency_label'].iloc[index])
            
    # creating dictionaries to count the frequencies
    pos_fn_dict = Counter(pos_false_negatives)
    lemma_fn_dict = Counter(lemma_false_negatives)
    dependency_fn_dict = Counter(dependency_labels_false_negatives)  
    
    return(pos_fn_dict, lemma_fn_dict, dependency_fn_dict)
    
def get_tp_dicts(filepath):
    """ 
    creates dictionaries to count how often aspects of true positives occur
    :param filepath: a string
    """
    
    df = pd.read_csv(filepath, sep='\t', header=0)
    
    # creating lists for the correctly predicted
    pos_correct = []
    lemma_correct = []
    dependency_label_correct = []
    
    # aspects
    for index, prediction in enumerate(df['predicted_cue_label']):
        if df.iloc[index]['cue_label'] == 0 and df.iloc[index]['predicted_cue_label'] == 1:
            pos_correct.append(df['POS'].iloc[index])
            lemma_correct.append(df['lemma'].iloc[index])
            dependency_labels_correct.append(df['dependency_label'].iloc[index])
            
    # dicts
    pos_correct_dict = Counter(pos_correct)
    lemma_correct_dict = Counter(lemma_correct)
    dependency_correct_dict = Counter(dependency_correct) 
    
    return (pos_correct_dict, lemma_correct_dict, dependency_correct_dict)

def get_totals(filepath):
    """
    returns total amount of false positives, false negatives and true positives of one corpus
    :param filepath: str
    """
    
    df = pd.read_csv(filepath, sep='\t', header=0)
    
    counter_fp = 0
    for index, prediction in enumerate(df['predicted_cue_label']):
        if df.iloc[index]['cue_label'] == 0 and df.iloc[index]['predicted_cue_label'] == 1:
            counter_fp += 1
            
    counter_fn = 0
    for index, prediction in enumerate(df['predicted_cue_label']):
        if df.iloc[index]['cue_label'] == 1 and df.iloc[index]['predicted_cue_label'] == 0:
            counter_fn += 1
            
    counter_tp = 0
    for index, prediction in enumerate(df['predicted_cue_label']):
        if df.iloc[index]['cue_label'] == 1 and df.iloc[index]['predicted_cue_label'] == 1:
            counter_tp += 1
            
    print('total of false positives: ' + str(counter_fp))
    print('total of false negatives: ' + str(counter_fn))     
    print('total of true positives: ' + str(counter_tp))
    
            
    