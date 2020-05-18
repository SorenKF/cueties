'''
This collection of functions is built to take a dataframe and output a list of instances for source-content classification.

This list of instances takes the form of:
[[source, content, label], ...]

'label' is 1 or 0, for whether source and content belong to the same gold attribution (these are the gold classifications)
'source' and 'content' are tuples of indices representing the location of these token spans within the df.

So if 'source' == (58, 59), then index 58 within the df contains the only element within this source.
   if 'content' == (57, 85), then the content is contained in the indices (57, 58, ..., 84) within the df.
   
To access the lines of the df with these entries, df.iloc[i1: i2] can be used.

Usage: collect_instances_main(df) is the main function to be used. This file is designed to be imported in full and then for
collect_instances_main(df) to be called to attain the list of instances for df.

'''

import pandas as pd
import time
from collections import defaultdict

def content_in_sentence(df):
    '''
    Takes a df with "attribution", "filename", and "sentence_number" columns and returns a list (column) of binary
    "sentence contains a content" labels
    '''
    sent_with_content = set()
    for filename, sentence_number, attribution in zip(df["filename"], df["sentence_number"], df["attribution"]):
        for att in attribution.split(" "):
            att_split = att.split("-")
            if att_split[0] not in {"_", "0", ""} and att_split[1] == "CONTENT":
                sent_with_content.add((filename, sentence_number))
    labels = []
    for filename, sentence_number in zip(df["filename"], df["sentence_number"]):
        if (filename, sentence_number) in sent_with_content:
            label = 1
        else:
            label = 0
        labels.append(label)
    return labels

def candidate_sources(df):
    '''
    Takes a df and returns a list of IOB labels for candidate sources
    
    Requires columns POS, relevant_ne, ne_info, content_in_sentence
    '''
    labels = []
    source_pos = {"NN", "NNP", "PRP", "PRP$", "NNS", "NNPS"}
    zipped = zip(df["POS"], df["relevant_ne"], df["ne_info"], df["content_in_sentence"])
    for pos, relevant_ne, ne_info, content_in_sentence in zipped:
        label = "O"
        if content_in_sentence == 0:
            labels.append(label)
            continue
        elif relevant_ne == 1:
            ne_IOB = ne_info.split("-")[0]
            if ne_IOB == "E" or ne_IOB == "I":
                label = "I"
            else: #ne_IOB=="B" or "S"
                label = "B"
        elif pos in source_pos:
            label = "B"
        labels.append(label)
    return labels

def collect_span_dict(df, att_part):
    '''
    Extracts a dictionary {(filename, sent_num): [content1, ...], ...} from the "attribution" column of a df.
    Requires columns filename, sentence_number, and attribution.
    
    The lists that are the values of the dictionary are lists of index tuples representing the location of the
    att_part in the df. The lengths of these lists are the number of attributions in the file (specified by 'filename')
    that the att_part appears in; this means that a SOURCE and a CONTENT dictionary extracted by this function
    will correspond in that the tuple content_i in position i of the list will belong to the same attribution as source_i
    in the same (filename, sent_num) location.
    
    :param df: a pandas df object
    :param att_part: "CONTENT" or "SOURCE"
    
    :returns return_dict: a dictionary as specified above
    '''
    current_sentence = 1
    current_filename = df["filename"][0]
    zipper = zip(range(len(df["filename"])), df["sentence_number"], df["attribution"], df["filename"])
    
    # These dictionaries are of the form {(filename, sentence_num): [content1, ...]}
    # With content1 = (start_index, end_index)
    # with len(contents) = num_attributions
    return_dict = dict()
    
    contents = [None] * len(df["attribution"][0].split(" "))
    
    for index, sentence, attribution, filename in zipper:
        if sentence != current_sentence or current_filename != filename:
            if check_for_contents(contents):
                for i in range(len(contents)):
                    if type(contents[i]) == int:
                        contents[i] = contents[i],index
                return_dict[(current_filename, current_sentence)] = contents
            current_sentence = sentence
            current_filename = filename
            contents = [None] * len(attribution.split(" "))
        att_list = attribution.split(" ")
        for i in range(len(att_list)):
            att_split = att_list[i].split("-")
            if att_split[0] in {"_", "0", ""} or att_split[1] != att_part or att_split[2] == "NE":
                # Outside of span for attribution i
                if type(contents[i]) == int:
                    # Have a start index for att i
                    contents[i] = contents[i], index
            else:
                if att_split[1] == att_part and att_split[0] == "B":
                    # Beginning of attribution i
                    contents[i] = index
    
    if check_for_contents(contents):
        for i in range(len(contents)):
            if type(contents[i]) == int:
                contents[i] = contents[i],index
        return_dict[(current_filename, current_sentence)] = contents

    return return_dict

def check_for_contents(l):
    '''
    Checks to see if a list l contains anything other than None objects
    '''
    for entry in l:
        if entry != None:
            return True
    return False

def collect_candidate_sources(df):
    '''
    Outputs a dictionary of of {(filename, sent_num): [cand_source, ...]}
    with cand_source in the form of index tuple
    '''
    current_sentence = 1
    current_filename = df["filename"][0]
    zipper = zip(range(len(df["filename"])), df["sentence_number"], df["candidate_source_label"], df["filename"])
    
    # These dictionaries are of the form {(filename, sentence_num): [content1, ...]}
    # With content1 = (start_index, end_index)
    # with len(contents) = num_attributions
    return_dict = dict()
    
    candidate_sources = []
    in_span = False
    start_index = 0
    
    for index, sentence, label, filename in zipper:
        #print(filename,sentence,label,in_span)
        if sentence != current_sentence or current_filename != filename:
            if candidate_sources:
                return_dict[(current_filename, current_sentence)] = candidate_sources
            current_sentence = sentence
            current_filename = filename
            candidate_sources = []
        if in_span:
            if label == "O":
                # If in_span, label 'should' be I or O, so we continue or close span, respectively
                # If label is B, we treat it as one continuous span, instead of 2 separate ones
                candidate_sources.append((start_index, index))
                in_span = False
        else:
            assert label != "I"
            if label == "B":
                start_index = index
                in_span = True
    if candidate_sources:
        return_dict[(current_filename, current_sentence)] = candidate_sources
    
    return return_dict

def collect_instances(content_dict, source_dict, cand_source_dict):
    '''
    Takes three dictionaries of {(filename, sent_num): [index_tuple, ...], ...}
    and returns a list of [(source), (content), label] representing instances for a source, content
    classifier. source and content are tuples representing spans, label is a binary label for whether
    source and content are attributed together
    '''
    return_list = []
    # First make dict of {location: [(content, source), ...]} pairs. If no source for a content, (source)==None
    gold_pairs = defaultdict(list)
    # Then make function to compare spans for overlap (get label from two source spans)
    for location, content_list in content_dict.items():
        if location in source_dict:
            source_list = source_dict[location]
        else:
            source_list = [None] * len(content_list)
        assert len(content_list)==len(source_list)
        
        for i in range(len(content_list)):
            content_span = content_list[i]
            if content_span != None:
                source_span = source_list[i]
                gold_pairs[location].append((source_span, content_span))
    
    for location, pair_list in gold_pairs.items():
        if location in cand_source_dict:
            cand_source_list = cand_source_dict[location]
        else:
            cand_source_list = []
        for source, content in pair_list:
            for cand_source in cand_source_list:
                if source != None and compare_spans(source, cand_source):
                    return_list.append([cand_source, content, 1])
                else:
                    return_list.append([cand_source, content, 0])
    return return_list
    
def compare_spans(span1, span2):
    '''
    Given two tuples of integers, determines whether there'a any overlap between them
    '''
    for i in range(span1[0], span1[1]):
        if span2[0] <= i and i < span2[1]:
            return True
    return False


def collect_instances_main(df):
    '''
    This is the main function. It takes a dataframe df and returns a list of instances, as explained in the
    documentation for this file.
    '''
    
    df["content_in_sentence"] = content_in_sentence(df)
    df["candidate_source_label"] = candidate_sources(df)
    cand_source_dict = collect_candidate_sources(df)
    content_dict = collect_span_dict(df, "CONTENT")
    source_dict = collect_span_dict(df, "SOURCE")
    
    instances = collect_instances(content_dict, source_dict, cand_source_dict)
    
    return instances
    