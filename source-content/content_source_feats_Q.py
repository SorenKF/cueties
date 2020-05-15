def create_instance_list(list_of_tuples, df):
    main_list = list()

    # Loop through instances meaning content and source span indices
    for instance in pair_list:
        # initiate instance_list and index_list
        attribution_indices = list()

        source, content = instance
        b_source, e_source = source
        b_content, e_content = content

        # For index of source
        for index in range(b_source, e_source + 1):
            attribution_indices.append(index)
        # for index of content
        for index in range(b_content, e_content + 1):
            attribution_indices.append(index)

        # Find gap indices
        if b_source < b_content:
            # If content follows source, gap is between last token of source and first of content
            gap_indices = [e_source, b_content]
        else:
            # If source follows content, then visa versa
            gap_indices = [e_content, b_source]

        list_tuple = (attribution_indices, gap_indices)
        main_list.append(list_tuple)

    return main_list


def find_sc_dist(df, main_list):
    df['s/c_distance'] = 'X'

    for list_tupel in main_list:
        instance_list = list_tupel[0]
        index_list = list_tupel[1]

        if instance_list[0] < instance_list[1]:
            distance = instance_list[1] - instance_list[0]
        else:
            distance = instance_list[0] - instance_list[1]

        df.loc[index_list, 's/c_distance'] = distance

    # return df


def find_num_conts_between(df, main_list):
    df['num_conts_between'] = 'X'

    for list_tupel in main_list:
        instance_list = list_tupel[0]
        index_list = list_tupel[1]

        counter = 0
        for index in list(range(instance_list[0], instance_list[1])):
            if df.loc[index, 'content_label_gold'] == 'B':
                counter += 1

        df.loc[index_list, 'num_conts_between'] = counter

    # return df


def find_num_sources_between(df, main_list):
    df['num_sources_between'] = 'X'

    for list_tupel in main_list:
        instance_list = list_tupel[0]
        index_list = list_tupel[1]

        counter = 0
        for index in list(range(instance_list[0], instance_list[1])):
            if df.loc[index, 'candidate_source'] == 'B':
                counter += 1

        df.loc[index_list, 'num_sources_between'] = counter

    # return df


def run_dist_feats(list_of_tuples, df):
    main_list = create_instance_list(list_of_tuples, df)
    find_sc_dist(df, main_list)
    find_num_conts_between(df, main_list)
    find_num_sources_between(df, main_list)

    return df