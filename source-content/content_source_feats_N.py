import pandas as pd
import instance_generation
import winsound
import time
from collections import defaultdict
# Change the filepaths in the main() function!!

def seperate_instance_gold(instance_output):
    gold_label_list = list()
    pair_list = list()
    for instance_list in instance_output:
        gold_label_list.append(instance_list[2])
        pair_tuple = (instance_list[0], instance_list[1])
        pair_list.append(pair_tuple)

    return pair_list, gold_label_list


def create_instance_list(pair_list, df):
    main_attribution = list()
    main_gap = list()

    # Loop through instances meaning content and source span indices
    for instance in pair_list:
        # initiate instance_list and index_list
        attribution_indices = list()
        source, content = instance
        b_source, e_source = source
        b_content, e_content = content

        # For index of source
        for index in range(b_source, e_source):
            attribution_indices.append(index)
        # for index of content
        for index in range(b_content, e_content):
            attribution_indices.append(index)

        # Find gap indices
        
        if e_source < b_content:
            # If content follows source, gap is between last token of source and first of content
            gap_indices = [e_source, b_content]
        elif e_content < b_source:
            # If source follows content, then visa versa
            gap_indices = [e_content, b_source]
        # If overlap, fill gap_indices with 0
        else:
            gap_indices = [0, 0]

        main_attribution.append(attribution_indices)
        main_gap.append(gap_indices)

    return main_attribution, main_gap


def get_word_length(pair_df, list_of_tuples):
    """
    Get word length for the content or source in the tuple
    """

    content_len_list = []
    source_len_list = []

    for s_tuple, c_tuple in list_of_tuples:
        # Get content len
        start, end = c_tuple
        content_len_list.append(end + 1 - start)

        # Get source len
        start, end = s_tuple
        source_len_list.append(end + 1 - start)
    pair_df['content_length'] = content_len_list
    pair_df['source_length'] = source_len_list


def check_s_in_c(token_df, pair_df, list_of_tuples):
    results = []
    for source_indices, content_indices in list_of_tuples:
        b_content, e_content = content_indices
        b_source, e_source = source_indices
        
        span_lemmas = " ".join(token_df.iloc[b_source:e_source + 1]["lemma"])
        content_lemmas = " ".join(token_df.iloc[b_content:e_content + 1]["lemma"])
        
        if span_lemmas in content_lemmas:
            result = 0
        else:
            result = 1
        results.append(result)

    pair_df['source_in_content'] = results


def get_distance_c2sentstart(token_df, pair_df, list_of_tuples):
    distances_2docstart = []
    distances_2sentstart = []
    distances_dict = defaultdict(list)

    i = 0 #DEBUG
    print(f"Total pairs: {len(list_of_tuples)}")
    for source_indices, content_indices in list_of_tuples:
        i += 1 #DEBUG
        if i % 1000 == 0:
            print(f"Finished {i} pairs...")
        if content_indices in distances_dict:
            # Found values for this content already, so use them and continue to next
            distance_c2doc = distances_dict[content_indices][0]
            distances_2docstart.append(distance_c2doc)

            distance_c2sent = distances_dict[content_indices][1]
            distances_2sentstart.append(distance_c2sent)

            continue

        b_content, e_content = content_indices

        # Get the distance from the start of the document to the content span.
        doc_filename = token_df.iloc[b_content]['filename']
        index_doc_start = b_content
        while token_df.iloc[index_doc_start]['filename'] == doc_filename:
            # This while loop decreases index_doc_start one index at a time, starting at b_content, until prev file is found
            if index_doc_start == 0:
                break
            else:
                index_doc_start -= 1
        # index_doc_start is now the last index in the previous file
        if index_doc_start != 0:
            index_doc_start += 1
        
        distance_c2doc = b_content - index_doc_start
        distances_dict[content_indices].append(distance_c2doc)
        distances_2docstart.append(distance_c2doc)

        # Get the distance from the start of the sentence to the content span.
        sent_id = token_df.iloc[b_content]["sentence_number"]
        index_sent_start = b_content
        while token_df.iloc[index_sent_start]['sentence_number'] == sent_id:
            # This while loop decreases index_sent_start one index at a time, starting at b_content, until prev sent is found
            if index_sent_start == 0:
                break
            else:
                index_sent_start -= 1
        # index_sent_start is now the last index in the previous sentence
        if index_doc_start != 0:
            index_doc_start += 1

        distance_c2sent = b_content - index_sent_start
        distances_dict[content_indices].append(distance_c2sent)
        distances_2sentstart.append(distance_c2sent)

    pair_df['distance_c2docstart'] = distances_2docstart
    pair_df['distance_c2sentstart'] = distances_2sentstart


def find_sc_dist(pair_df):
    pair_df['s/c_distance'] = 'X'

    dist_list = list()

    for b_gap, e_gap in pair_df['gap_indices']:

        if b_gap == e_gap == 0:
            distance = 0
        elif b_gap < e_gap:
            distance = e_gap - b_gap
        else:
            distance = b_gap - e_gap

        dist_list.append(distance)

    pair_df['s/c_distance'] = dist_list


def find_num_conts_between(token_df, pair_df):
    pair_df['num_conts_between'] = 'X'
    count_list = list()

    for b_gap, e_gap in pair_df['gap_indices']:
        if b_gap == e_gap == 0:
            counter = 0
            count_list.append(counter)
        else:
            counter = 0
            for index in range(b_gap, e_gap):
                if token_df.loc[index, 'content_label_gold'] == 'B':
                    counter += 1
            count_list.append(counter)

    pair_df['num_conts_between'] = count_list


def get_pairwise_data(filepath):
    # Read in token level data
    token_df = pd.read_csv(filepath, delimiter='\t', index_col=0)
    #token_df = token_df[:500]

    instance_output = instance_generation.collect_instances_main(token_df)
    print("Got instances.")
    # Extract relevant lists
    # Pair_list is list of tuples of source and content start and end indices
    # Gold_labels is list of labels whether source and content are a match
    # Sequence_indices is a list of lists of all the indicis which make up a sequence of soure-content
    # Gap_indices is a list of start and end indices of the gap between the source and the content
    pair_list, gold_labels = seperate_instance_gold(instance_output)
    sequence_indices, gap_indices = create_instance_list(pair_list, token_df)

    # return pair_list, gold_labels, sequence_indices, gap_indices

    assert len(pair_list) == len(gold_labels) == len(sequence_indices) == len(gap_indices), f'Lengths \
    do not match. {len(pair_list)}, {len(gold_labels)}, {len(sequence_indices)}, {len(gap_indices)}.'

    # Create DataFrame to contain pairwise data
    pair_df = pd.DataFrame()
    pair_df['source_content_boundaries'] = pair_list
    pair_df['gold_labels'] = gold_labels
    pair_df['all_indices_in_span'] = sequence_indices
    pair_df['gap_indices'] = gap_indices

    # Add features to dataframe
    print("Getting word length...")
    get_word_length(pair_df, pair_df['source_content_boundaries'])
    print("Getting distance c2sentstart...")
    get_distance_c2sentstart(token_df, pair_df, pair_df['source_content_boundaries'])
    print("Checking for s_in_c...")
    check_s_in_c(token_df, pair_df, pair_df['source_content_boundaries'])
    print("Finding sc_dist...")
    find_sc_dist(pair_df)
    print("Finding contents between...")
    find_num_conts_between(token_df, pair_df)

    pair_df.drop(['all_indices_in_span', 'gap_indices'], axis=1)

    return pair_df

def main():
    print('starting')
    test_filepath = "Data/source-content/polnear_features/polnear_dev_features.tsv"
    test_fileout = "Data/source-content/polnear_dev_pairs.tsv"
    train_filepath = "Data/source-content/polnear_features/polnear_train_features.tsv"
    train_fileout = "Data/source-content/polnear_train_pairs.tsv"

    start = time.time()
    
    print(f'reading in {test_filepath}')
    df_test = get_pairwise_data(test_filepath)
    print('found pairwise data')
    df_test.to_csv(test_fileout, sep='\t')
    print(f'written to {test_fileout}')
    print(f"Time elapsed: {time.time()-start}")
    
    
    print(f'reading in {train_filepath}')
    df_train = get_pairwise_data(train_filepath)
    print('found pairwise data')
    df_train.to_csv(train_fileout, sep='\t')
    print(f'written to {train_fileout}')
    print(f"Time elapsed: {time.time()-start}")
    

if __name__ == '__main__':
    main()
