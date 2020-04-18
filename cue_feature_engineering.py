import pandas as pd
import numpy
import glob

def prep_df(df):
    df = pd.read_csv(filepath, delimiter='\t', index_col=0)
    df[['sentence_number', 'doc_token_number', 'sentence_token_number', 'token', 'lemma', 'POS', 
    'dependency_head', 'dependency_label', 'ne_info',   
       'cue_label']]
    
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


    