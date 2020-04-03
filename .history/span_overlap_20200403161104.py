# Here's code for using main_utils in a notebook:
#import pandas as pd
import os
from main_utils import import_attribution_doc, extract_attributions, extract_attribution_spans

# here's a sound snippet to play a beep once this MF has run. The call to play the soudn is at one of the last lines.
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
####

# Get span of attributions


# Check if attribute spans multiple sentences and count
def count_span_sentence_overlaps(df, attribution_spans):
    """
    Counts the amount of attribution which span one single sentence and those that span
    multiple sentences.
    Returns: tuple containing count of attributions which span one sentence and those that span
    multiple.
    """
    one_sentence_span = 0
    multiple_sentence_span = 0
    for start_index, end_index in attribution_spans:
        if start_index == 9999999 or end_index == 0:
            print(df.at[0, 'filename'])  # These prints are for 
            print(start_index)
            print(end_index)
            continue
        if df.at[start_index, 'sentence_number'] == df.at[end_index, 'sentence_number']:
            one_sentence_span += 1
        else:
            multiple_sentence_span += 1
    return one_sentence_span, multiple_sentence_span


def main():
    # Replace with your path (obvs)
    parc_directory = "./../Data/parc30-conll/train-conll-foreval/"
    polnear_directory = "./../Data/polnear-conll/train-conll-foreval/" # remember the folder structure should be ./../Data/corpus/corpus_subset/corpus_file1.xml

    one_sentence_total = 0
    multiple_sentences_total = 0

    i = 1
    for filename in os.listdir(polnear_directory): #specify which dir you want to run the code on (i.e. which corpus from above). Adjust on line 53 accordingly.
        if i % 50 == 0:
            # This bit just lets you know where you are (prints some stuff every 100 files)
            print(filename)
            print('one sentence:', one_sentence_total, 'multiple sentence:', multiple_sentences_total)
        i += 1
        df = import_attribution_doc(polnear_directory + filename)
        if df["attribution"][0] != 0:
            atts = extract_attributions(df)
            att_spans = extract_attribution_spans(atts)
            one_sentence, multiple_sentences = count_span_sentence_overlaps(df, att_spans)
            one_sentence_total += one_sentence
            multiple_sentences_total += multiple_sentences
    print()
    print('one sentence:', one_sentence_total)
    print('multiple sentence:', multiple_sentences_total)


if __name__ == '__main__':
    main()

    # Just some signals that the script is done.
    print('DONE!')
    winsound.Beep(freq, duration)
