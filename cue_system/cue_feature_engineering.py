import pandas as pd
import numpy
import glob

def prep_df(df):
    df = pd.read_csv(filepath, delimiter='\t', index_col=0)
    df[['sentence_number', 'doc_token_number', 'sentence_token_number', 'token', 'lemma', 'POS', 
    'dependency_head', 'dependency_label', 'ne_info',   
       'cue_label']]
    
    return df

def get_previous_or_following(df, column_name, step=-1):
    """
    Gets previous or following label, for instance pos tag or token.

    :param df: Dataframe to apply to
    :param column_name str: column to apply to
    :param step integer: -1 for previous, 1 for next, -2 for one before previous...

    :returns df:
    """
    for i in range(df.shape[0]):
        if step == 0:
            break
        # If the previous or following row is not in the range of the df
        # then take '.' as value
        if (i + step < 0) or (i + step >= df.shape[0]):
            value = '.'
        # Otherwise take the item at i+step
        else:
            value = df.at[i + step, column_name]
        if step > 0:
            step_str = f'+{step}'
        else:
            step_str = str(step)
        # Fill in df
        df.at[i, f'{column_name}_{step_str}'] = value
        
def add_tokens_window5(df):
    
    get_previous_or_following(df, 'token', step=-1)
    get_previous_or_following(df, 'token', step=-2)
    get_previous_or_following(df, 'token', step=-3)
    get_previous_or_following(df, 'token', step=-4)
    get_previous_or_following(df, 'token', step=-5)

    get_previous_or_following(df, 'token', step=1)
    get_previous_or_following(df, 'token', step=2)
    get_previous_or_following(df, 'token', step=3)
    get_previous_or_following(df, 'token', step=4)
    get_previous_or_following(df, 'token', step=5)

    return df

def add_lemmas_window5(df):
    
    get_previous_or_following(df, 'lemma', step=-1)
    get_previous_or_following(df, 'lemma', step=-2)
    get_previous_or_following(df, 'lemma', step=-3)
    get_previous_or_following(df, 'lemma', step=-4)
    get_previous_or_following(df, 'lemma', step=-5)

    get_previous_or_following(df, 'lemma', step=1)
    get_previous_or_following(df, 'lemma', step=2)
    get_previous_or_following(df, 'lemma', step=3)
    get_previous_or_following(df, 'lemma', step=4)
    get_previous_or_following(df, 'lemma', step=5)

    return df

def add_pos_window5(df):
    
    get_previous_or_following(df, 'POS', step=-1)
    get_previous_or_following(df, 'POS', step=-2)
    get_previous_or_following(df, 'POS', step=-3)
    get_previous_or_following(df, 'POS', step=-4)
    get_previous_or_following(df, 'POS', step=-5)

    get_previous_or_following(df, 'POS', step=1)
    get_previous_or_following(df, 'POS', step=2)
    get_previous_or_following(df, 'POS', step=3)
    get_previous_or_following(df, 'POS', step=4)
    get_previous_or_following(df, 'POS', step=5)

    return df

def add_bigrams_prev(df):
    
    df['bigram_prev_token'] = df['token'] + ' ' + df['token_-1']
    df['bigram_prev_lemma'] = df['lemma'] + ' ' + df['lemma_-1']
    df['bigram_prev_POS'] = df['POS'] + ' ' + df['POS_-1']
    
    return df

def add_bigrams_following(df):
    
    df['bigram_following_token'] = df['token'] + ' ' + df['token_+1']
    df['bigram_following_lemma'] = df['lemma'] + ' ' + df['lemma_+1']
    df['bigram_following_POS'] = df['POS'] + ' ' + df['POS_+1']
    
    return df

def shape(token):
    '''
    Takes token (str), and returns str of shape.

    Get short shape of token.
    lower = x
    upper = X
    digit = d
    other = o
    i.e
    cats -> x
    Cats -> Xx
    USoA -> XxX
    1999 -> d
    13dec19 -> dxd
    U.S.A -> XoXoXo
    '''

    # Create empty list to store shape symbols in
    shape_list = []

    # To prevent breaking on NaN values
    if type(token) != str:
        shape = 'o'
        return shape

    # Loop over every character
    for character in token:
        # If token is NaN


        # For any character except for first (swapped with other if statement for faster computing)
        if len(shape_list) > 0:
            # If the character is upper case, and the previous shape symbol is not upper,
            # set shape symbol to 'X'
            if character.isupper() and shape_list[-1] != 'X':
                shape_character = 'X'
            elif character.islower() and shape_list[-1] != 'x':
                shape_character = 'x'
            elif character.isdigit() and shape_list[-1] != 'd':
                shape_character = 'd'
            # If character is not upper, lower or digit (and the previous symbol is 'o')
            elif not any([character.isupper(), character.islower(), character.isdigit(),
                         shape_list[-1] == 'o']):
                shape_character = 'o'
            # If not the case (ie previous was upper and so is this one), continue to next character
            else:
                continue

        # For first character
        else:
            if character.isupper():
                shape_character = 'X'
            elif character.islower():
                shape_character = 'x'
            elif character.isdigit():
                shape_character = 'd'
            elif not any([character.isupper(), character.islower(), character.isdigit()]):
                shape_character = 'o'
            else:
                continue

        shape_list.append(shape_character)

    shape = ''.join(shape_list)
    return shape

def add_shape(df):
    
    df['shape'] = df.apply(lambda row:
                          shape(row['token']), axis=1)
    return df

def add_relevant_ne(df):
    
    relevant_ne = ['PERSON', 'ORG', 'GPE', 'LOC', 'NORP', 'FAC']

    useful_ne = list()
    df['ne_short'] = df.apply(lambda row: row['ne_info'][2:], axis=1)

    for ne in relevant_ne:
        useful_ne += list(df.loc[df['ne_short']== ne].index)

    #print(useful_ne)
    df['relevant_ne'] = 0
    df.loc[useful_ne,'relevant_ne'] = 1

    return df

def add_ne_info_window5(df):
    
    ne_indices = list(df[df['relevant_ne'] == 1].index)

    ne_set= set()

    for index in ne_indices:
        ne_set.add(index)
        ne_set.add(index-1)
        ne_set.add(index-2)
        ne_set.add(index-3)
        ne_set.add(index-4)
        ne_set.add(index-5)
        ne_set.add(index+1)
        ne_set.add(index+2)
        ne_set.add(index+3)
        ne_set.add(index+4)
        ne_set.add(index+5)

    #print(ne_set)

    ne_list= list()
    for index in ne_set:
        if index in range(0, len(df.index)):
            ne_list.append(index)
        else:
            continue

    df['ne_+-5'] = 0
    df.loc[ne_list, f'ne_+-5'] = 1
    df['ne_+-5'] = df['ne_+-5'].astype('int64')

    return df

def add_lexicon_check(df, lexicon):
    '''
    Takes a pandas df with at least a "token" column and a lexicon (a set of tokens) and adds a column 
    to the df indicating for each token in the df whether it is in the lexicon or not
    
    :param df: a pandas dataframe with at least a "token" column
    :param lexicon: a set of words
    '''
    in_lexicon = []
    for token in df["token"]:
        if token in lexicon:
            in_lexicon.append(1)
        else:
            in_lexicon.append(0)
    df["lexicon_check"] = in_lexicon
    
    return df


def add_quotation(df):
    
    df['quotation'] = 'O'
    
    qmark_b_indices = list()
    qmark_e_indices = list()
    
  
    qmark_b_indices += list(df.loc[df['lemma']== '``' ].index)
    qmark_e_indices += list(df.loc[df['lemma']== '‘‘'].index)
        
    zipped_indices = zip(qmark_b_indices, qmark_e_indices)
    b_e_indices = list(zipped_indices)
    
    for b,e in b_e_indices:
        #for b, e in pair:
        span = range(b, e)
        df.loc[span,'quotation'] = 'I'
        df.loc[b, 'quotation'] = 'B'
        df.loc[e, 'quotation'] = 'E'
            
    return df


def get_boundary_indices(df, boundary_type='sent'):
    """
    Gets indices of sentence or doc boundaries. For sentence boundary fill in 'sent' (is default). For doc fill in anything else.
    returns list of indices.
    """
    indices = set()
    # Boundary at start of doc for sent and doc
    indices.add(0)
    # If set to sentence boundaries then find all starts of sentences
    if boundary_type == 'sent':
        for index in df[df['sentence_token_number'] == 1].index:
            if index >= 2:   # Only if index is not 0 then add index to list
                indices.add(index-2)
                indices.add(index-1)
                indices.add(index)
            elif index == 1:
                indices.add(index)
    
    # Also sentence boundary at end of doc
    indices.add(df.shape[0]-2)
    indices.add(df.shape[0]-1)
    return indices

def add_boundary_column(df, boundary_type):
    """
    Boundary type is 'sent' or 'doc'
    """
    df[f'near_{boundary_type}_boundary'] = 0
    indices = get_boundary_indices(df, boundary_type=boundary_type)
    df.loc[indices, f'near_{boundary_type}_boundary'] = 1
    
    return df
    
def get_sent_start_indices(df):
    indices = set()
    for index in df[df['sentence_token_number'] == 1].index:
        indices.add(index)
    return indices

def get_sent_bound_indices(df):
    sent_start_indices = get_sent_start_indices(df)
    sent_end_indices = set()
    
    for index in sent_start_indices:
        if index>0:
            sent_end_indices.add(index-1)
    sent_end_indices.add(df.shape[0]-1)
    sent_bound_indices = zip(sent_start_indices, sent_end_indices)
    
    return list(sent_bound_indices)

def add_sent_distance_metrics_to_df(df):
    df['dist_beg_sent'] = df['sentence_token_number'] -1
    sent_bound_indices = get_sent_bound_indices(df)
    for start_i, end_i in sent_bound_indices:
        range_indices = list(range(start_i, end_i+1))
        df.loc[range_indices, 'dist_end_sent'] = df.loc[end_i, 'sentence_token_number'] - df['dist_beg_sent'] -1
        df.loc[range_indices, 'sent_len'] = df.loc[end_i, 'sentence_token_number']
    df['dist_end_sent'] = df['dist_end_sent'].astype('int64')
    df['sent_len'] = df['sent_len'].astype('int64')
    
    return df

def add_in_sentence_bools_to_df(df):
    '''edited by Ellie'''
    sent_bound_indices = get_sent_bound_indices(df)
    df['pn_in_sent'] = 0
    df['ne_in_sent'] = 0
    df['qm_in_sent'] = 0
    for start_i, end_i in sent_bound_indices:
        range_indices = list(range(start_i, end_i+1))
        if df.loc[range_indices].loc[df['POS'] == 'PNP'].shape[0] != 0:
            df.loc[range_indices, 'pn_in_sent'] = 1
        if df.loc[range_indices].loc[df['relevant_ne'] == 1].shape[0] != 0:
            df.loc[range_indices, 'ne_in_sent'] = 1
        if df.loc[range_indices].loc[df['quotation'] != 'O'].shape[0] != 0:
            df.loc[range_indices, 'qm_in_sent'] = 1
    return df 

def add_any_in_sent(df):

    df['any_in_sent'] = 0
    item_indices = list()
    
    item_indices += list(df.loc[df['pn_in_sent']== 1].index)
    item_indices += list(df.loc[df['ne_in_sent']== 1].index)
    item_indices += list(df.loc[df['qm_in_sent']== 1].index)
    
    df.loc[item_indices,'any_in_sent'] = 1
            
    return df

def add_quotation_extra(df):
    
    df['pn_in_sent'] = df['pn_in_sent'].astype(str)
    df['ne_in_sent'] = df['ne_in_sent'].astype(str)
    df['qm_in_sent'] = df['qm_in_sent'].astype(str)
    
    
    
    df['quotation_pn'] = '_'
    df['quotation_pn'] = df['quotation'] + df['pn_in_sent']
    
    df['quotation_ne'] = '_'
    df['quotation_ne'] = df['quotation'] + df['ne_in_sent']
    
    df['quotation_qm'] = '_'
    df['quotation_qm'] = df['quotation'] + df['qm_in_sent']
            
    return df

def create_feature_df_cue(filepath):
    '''final main function combining all feature engineering steps'''
    
    clean = prep_df(df)
    a = add_tokens_window5(clean)
    b = add_lemmas_window5(a)
    c = add_pos_window5(b)
    d = add_bigrams_prev(c)
    e = add_bigrams_following(d)
    f = add_shape(e)
    g = add_relevant_ne(f)
    h = add_ne_info_window5(g)
    i = add_lexicon_check(h, lexicon)
    j = add_quotation(i)
    k = add_boundary_column(j, boundary_type='sent')
    l = add_boundary_column(k, boundary_type='doc')
    m = add_sent_distance_metrics_to_df(l)
    n = add_in_sentence_bools_to_df(m)
    o = add_any_in_sent(n)
    p = add_quotation_extra(o)
    
    final_df = p
    
    return final_df