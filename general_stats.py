import pandas as pd
import os
from main_utils import import_attribution_doc
from statistics import mean


def get_corpus_stats(corpus_directory):
    """
    Function to find the general statistics on each file in a corpus and in the corpus overall.

    takes one argument: corpus directory- the relative path to the directory containing the corpus.

    returns 2 items: main_list- a list of dictionaries with each dictionary containing stats for each file in the corpus
                    stats_dict- a dictionary containing the stats for the overall corpus
    :param corpus_directory:
    """

    main_list = list()
    for filename in os.listdir(corpus_directory):

        df = import_attribution_doc(corpus_directory + filename)

        doc_dict = dict()

        for col in df.columns:
            filename = df['filename'][0]
            sens = df['sentence_number'].max()

            sent_lens = list()
            count = 0
            for row in df['sent_token_number']:
                count += 1
                if row == 1:
                    sent_lens.append(count)
                    count = 0
            last_item = df['sent_token_number'].iloc[-1]
            sent_lens.append(last_item)
            av_len = mean(sent_lens[1:])

            token_count = df['doc_token_number'].iloc[-1]

        doc_dict['filename'] = filename
        doc_dict['number of sentences'] = sens
        doc_dict['average sentence length'] = av_len
        doc_dict['number of tokens'] = token_count

        main_list.append(doc_dict)

    # print(main_list)

    stats_dict = dict()

    sent_count = list()
    token_count = list()
    sent_len_count = list()

    for doc_dict in main_list:
        sent_count.append(doc_dict['number of sentences'])
        token_count.append(doc_dict['number of tokens'])
        sent_len_count.append(doc_dict['average sentence length'])

    av_sent_count = mean(sent_count)
    av_token_count = mean(token_count)
    av_sent_len = mean(sent_len_count)

    stats_dict['number of docs'] = len(main_list)
    stats_dict['average number of sentences'] = av_sent_count
    stats_dict['average number of tokens'] = av_token_count
    stats_dict['average sentence length'] = av_sent_len
    # print(stats_dict)

    return main_list, stats_dict
