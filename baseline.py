import os
from main_utils import import_attribution_doc

# Create baseline

# Get cue frequencies and make a common list.

def get_cue_frequencies(folder):
    """
    This function extracts all attributions in a corpus and returns
    - a dictionary with for every entry a string of lemmas in cue spans as key and frequency as value
    - a dictionary with for every entry a string of POS in cue spans as key and frequency as value
    """

    lemma_list_cue = []
    freq_dict_lemma_cue = {}
    pos_list_cue = []
    freq_dict_pos_cue = {}

    for filename in os.listdir(folder):
        data = import_attribution_doc(folder + filename)
        if data["attribution"][0] != 0:



folderpath_parc = "../parc30-conll/train-conll-foreval/"
folderpath_polnear = '../polnear30-conll/train-conll-foreval/'
get_cue_frequencies(folderpath_parc)
