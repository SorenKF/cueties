# Converts a directory of conll attribution files into new format.
# To run: python preprocessing_utils.py infolder outfolder

import stanza
import main_utils
import pandas as pd
import os
import sys
import time

def extract_cue_labels(attribution_df):
    '''
    Takes an attribution pandas df and returns the same df with an added column "cue_label"
    
    :param attribution_df: a pandas dataframe with conll attribution column format
    
    :returns new_df: the same df with an added column "cue_label": a binary label for whether each token is a cue
    '''
    new_df = attribution_df.copy()
    cue_info = [0]*len(new_df["word"])
    
    # Deal with case where file has no attributions; return immediately
    if attribution_df["attribution"][0] == 0:
        attribution_df["cue_label"] = cue_info
        return attribution_df
    
    # First, gather spans of all cues
    attribution_info = attribution_df["attribution"]
    num_cues = len(attribution_info[0].split(" "))
    cues = [] 
    attribution_part = "CUE"
    for i in range(num_cues):
        cues.append([])
    for att_index in range(num_cues):
        start_index = None
        for i in range(len(attribution_info)):
            attribution_string = attribution_info[i].split(" ")[att_index].split("-")
            if attribution_string != ["_"] and attribution_string[2] == "NE":
                # Skip nested attributions
                continue
            if attribution_string == ["_"] or attribution_string[1] != attribution_part:
                # Not inside a span
                if start_index != None:
                    # Found the end of a span we were in
                    cues[att_index].append((start_index, i))
                    start_index = None
            else:
                IOB_label, att_part = attribution_string[0], attribution_string[1]
                if att_part == attribution_part and IOB_label == "B":
                    start_index = i
                    
    for span_list in cues:
        last_index_in_cue = -1
        for span in span_list:
            if span[-1] - 1 > last_index_in_cue:
                last_index_in_cue = span[-1]-1
        if last_index_in_cue != -1:
            cue_info[last_index_in_cue] = 1
    new_df["cue_label"] = cue_info
    return new_df

def file_to_sents(attribution_df):
    '''
    Takes a conll format attribution pandas df and returns the text in the form of a list of
    sentences (which are lists of tokens)
    
    :param attribution_df: a pandas dataframe with conll attribution column format
    
    :returns sentences: a list of lists of tokens
    '''
    #cue_labels = extract_cue_labels(attribution_df)
    
    tokens = zip(attribution_df["sentence_number"], attribution_df["word"])
    sentences = []
    current_sentence_num = 1
    current_sentence = []
    current_token = 0 # for error message
    
    for sentence_number, token in tokens:
        current_token += 1
        if sentence_number == current_sentence_num:
            current_sentence.append(token)
        elif sentence_number == current_sentence_num + 1:
            sentences.append(current_sentence)
            current_sentence = [token]
            current_sentence_num = sentence_number
        else:
            # mis-ordered sentences or tokens, this is an error
            raise Exception(f"Order Error, line {current_token}")
    sentences.append(current_sentence) # last sentence
    return sentences
    
def stanza_to_df(doc):
    '''
    Converts a stanza doc object and a list of corresponding cue labels into a pandas df
    
    :param doc: a stanza doc object
    
    :returns df: a pandas dataframe with columns: sentence_number, doc_token_number, sentence_token_number, 
        token, lemma, POS, dependency_label, dependency_head, ne_info
    
    '''

    list_of_dicts = []
    position_in_doc = 0
    sent_number = 0
    
    for sent in doc.sentences:
        sent_number += 1
        for word, token in zip(sent.words, sent.tokens):
            position_in_doc += 1
            token_dict = dict()
            token_dict["sentence_number"] = sent_number
            token_dict["doc_token_number"] = position_in_doc
            token_dict["sentence_token_number"] = int(word.id)
            token_dict["token"] = word.text
            token_dict["lemma"] = word.lemma
            token_dict["POS"] = word.xpos
            token_dict["dependency_label"] = word.deprel
            token_dict["dependency_head"] = word.head
            token_dict["ne_info"] = token.ner
            list_of_dicts.append(token_dict)
          
    df = pd.DataFrame.from_dict(list_of_dicts)

    return df

def preprocessing_main(corpus_directory, new_directory):
    '''
    Main function for preprocessing: takes a corpus directory and converts each file to the
    new format (.tsv files from pandas dfs) with updated info (all columns predicted by stanza
    and new cue_label column)
    
    :param corpus_directory: the path to a folder containing all of the desires corpus files
    :param new_directory: the path to the folder where new files will be written
    
    '''
    start_time = time.time()
    
    nlp=stanza.Pipeline('en',tokenize_pretokenized=True)
    
    finished = 0
   
    for file in os.listdir(corpus_directory):
        if finished % 100 == 0:
            print(f"Finished {finished}. On to file {file}")
            print(f"Time elapsed: {time.time()-start_time}")
        outfile = file.split(".")[0] + ".tsv"
        
        df = main_utils.import_attribution_doc(corpus_directory+file)
        cue_label_df = extract_cue_labels(df)
    
        sents = file_to_sents(cue_label_df)
        doc = nlp(sents)
    
        stanza_df = stanza_to_df(doc)
        assert len(stanza_df)==len(cue_label_df["cue_label"]), f"Fatal error; file {file}"
        stanza_df["cue_label"] = cue_label_df["cue_label"]
        assert len(stanza_df) == len(cue_label_df["attribution"]), f"Fatal error; file {file}"
        stanza_df["attribution"] = cue_label_df["attribution"]
        
        stanza_df.to_csv(new_directory+outfile, sep="\t")
        
        finished += 1
    end_time = time.time()
    print(f"Total time elapsed: {end_time-start_time}")

def import_converted_file(filename):
    '''
    Takes a converted conll attribution file and returns as a pandas df
    
    '''
    df = pd.read_csv(filename, sep='\t', header=0)
    
    return df
    
if __name__ == "__main__":
    
    inp = sys.argv
    assert len(inp) == 3, "incorrect run command"
    
    infolder = inp[1]
    outfolder = inp[2]
    
    preprocessing_main(infolder, outfolder)