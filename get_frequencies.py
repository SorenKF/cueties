import pandas as pd
import os
from main_utils import import_attribution_doc, extract_attributions
import json

######### comment out and replace filpath to directory you want to use
directory = "C:/Users/Stell/Documents/Attribution System/polnear-conll/polnear-conll/train-conll-foreval/"


# directory = "C:/Users/Stell/Documents/Attribution System/parc30-conll/train-conll-foreval/"

def get_cue_frequencies():
    """
    This function extracts all attributions in a corpus and returns 
    - a dictionary with for every entry a string of lemmas in cue spans as key and frequency as value
    - a dictionary with for every entry a string of POS in cue spans as key and frequency as value
    """

    lemma_list_cue = []
    freq_dict_lemma_cue = {}
    pos_list_cue = []
    freq_dict_pos_cue = {}

    for filename in os.listdir(directory):
        df = import_attribution_doc(directory + filename)
        if df["attribution"][0] != 0:
            atts = extract_attributions(df)

            for i in range(len(atts)):
                attribution = atts[i]
                cue_span = attribution["CUE"]

                for span in cue_span:
                    if span == None:
                        continue
                    else:
                        cue = ""
                        word_list = df["lemma"][span[0]:span[1]]
                        for word in word_list:
                            cue += word + " "
                        lemma_list_cue.append(cue.strip(" "))

                        pos = ""
                        pos_list = df['POS'][span[0]:span[1]]
                        for tag in pos_list:
                            pos += tag + " "
                        pos_list_cue.append(pos.strip(" "))

    for item in lemma_list_cue:
        if item in freq_dict_lemma_cue:
            freq_dict_lemma_cue[item] += 1
        else:
            freq_dict_lemma_cue[item] = 1

    for item in pos_list_cue:
        if item in freq_dict_pos_cue:
            freq_dict_pos_cue[item] += 1
        else:
            freq_dict_pos_cue[item] = 1

    return (freq_dict_lemma_cue, freq_dict_pos_cue)


def get_source_frequencies():
    """
    This function extracts all attributions in a corpus and returns 
    - a dictionary with for every entry a string of lemmas in source spans as key and frequency as value
    - a dictionary with for every entry a string of POS in source spans as key and frequency as value
    """

    lemma_list_source = []
    freq_dict_lemma_source = {}
    pos_list_source = []
    freq_dict_pos_source = {}

    for filename in os.listdir(directory):
        df = import_attribution_doc(directory + filename)
        if df["attribution"][0] != 0:
            atts = extract_attributions(df)

            for i in range(len(atts)):
                attribution = atts[i]
                source_span = attribution["SOURCE"]

                for span in source_span:
                    if span == None:
                        continue
                    else:
                        source = ""
                        word_list = df["lemma"][span[0]:span[1]]
                        for word in word_list:
                            source += word + " "
                        lemma_list_source.append(source.strip(" "))

                        pos = ""
                        pos_list = df['POS'][span[0]:span[1]]
                        for tag in pos_list:
                            pos += tag + " "
                        pos_list_source.append(pos.strip(" "))

    for item in lemma_list_source:
        if item in freq_dict_lemma_source:
            freq_dict_lemma_source[item] += 1
        else:
            freq_dict_lemma_source[item] = 1

    for item in pos_list_source:
        if item in freq_dict_pos_source:
            freq_dict_pos_source[item] += 1
        else:
            freq_dict_pos_source[item] = 1

    return (freq_dict_lemma_source, freq_dict_pos_source)


def main():
    lemmas_cue = get_cue_frequencies()[0]
    pos_cue = get_cue_frequencies()[1]
    lemmas_source = get_source_frequencies()[0]
    pos_source = get_source_frequencies()[1]

    print(lemmas_cue)
    print()
    print(pos_cue)
    print()
    print(lemmas_source)
    print()
    print(pos_source)


#     for key, value in lemmas_cue.items():
#         if value > 15:
#             print(key, value)
#     print()
#     for key, value in pos_cue.items():
#         if value > 5:
#             print(key, value)
#     print()
#     for key, value in lemmas_source.items():
#         if value > 3:
#             print(key, value)
#     print()
#     for key, value in pos_source.items():
#         if value > 3:
#             print(key, value)

if __name__ == "__main__":
    main()
