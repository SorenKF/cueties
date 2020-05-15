def create_content_source_dist_feats(list_of_tuples, df):
    '''A function to add all content/source distance features to the df
    takes- list_of_tuples = a list of tuples of tuples of indices of the paired source/content spans
         - df = the dataframe containing the dataset
    returns- the df with added s/c_distance, num_conts_between and num_sources_between columns. Where X is used for tokens
    outside of the s/c pairs and numbers are given to all tokens inside of the s/c span to show the information in column'''
    
    df['s/c_distance'] = 'X'
    df['num_conts_between'] = 'X'
    df['num_sources_between'] = 'X'
    
    for instance in list_of_tuples:
        instance_list = list()
        index_list = list()
        
        source = instance[0]
        b1 = source[0]
        e1 = source[1]
        index_list.append(e1)
        for item in range(b1, e1):
            index_list.append(item)
        
        if e1 > b1:
            instance_list.append(e1)
        else:
            instance_list.append(b1)
        
        content = instance[1]
        b2 = content[0]
        e2 = content[1]
        index_list.append(e2)
        for item in range(b2, e2):
            index_list.append(item)
        
        for index in instance_list:
            if index < b2:
                if b2 < e2:
                    instance_list.append(b2)
                else:
                    instance_list.append(e2)
            elif index > b2:
                if b2 < e2:
                    instance_list.append(e2)
                else:
                    instance_list.append(b2)
        
        if instance_list[0] < instance_list[1]:
            distance = instance_list[1] - instance_list[0]
        else:
            distance = instance_list[0] - instance_list[1]
   
        df.loc[index_list,'s/c_distance'] = distance
    
        counter = 0
        for index in list(range(instance_list[0], instance_list[1])):
            if df.loc[index,'content_label_gold'] == 'B':
                counter += 1
                
        df.loc[index_list,'num_conts_between'] = counter
        
        counter = 0
        for index in list(range(instance_list[0], instance_list[1])):
            if df.loc[index,'candidate_source'] == 'B':
                counter += 1
                
        df.loc[index_list,'num_sources_between'] = counter
        
    return df