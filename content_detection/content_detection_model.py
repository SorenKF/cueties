# If you have not done so already: pip install sklearn-crfsuite
import sklearn_crfsuite
import pandas as pd
import glob
from csv import DictReader

filename = './../../Data/parc_features/parc_dev_features.tsv'

with open(filename, 'r') as infile:
    dict_reader = DictReader(infile, delimiter='\t')
    list_of_dict = list(dict_reader)

print('running')

for row in list_of_dict[:5]:
    print(row['dependency_label'])

print('ran')