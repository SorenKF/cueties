import pandas as pd
import instance_generation

def create_instance_list(list_of_tuples,df):
    
    main_list = list()
    
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
                    
        list_tupel = (instance_list, index_list)
        main_list.append(list_tupel)
    
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

        df.loc[index_list,'s/c_distance'] = distance

    #return df
    
def find_num_conts_between(df, main_list):
    
    df['num_conts_between'] = 'X'
    
    for list_tupel in main_list:
        instance_list = list_tupel[0]
        index_list = list_tupel[1]
        
        counter = 0
        for index in list(range(instance_list[0], instance_list[1])):
            if df.loc[index,'content_label_gold'] == 'B':
                counter += 1
                
        df.loc[index_list,'num_conts_between'] = counter
        
    #return df
    
def find_num_sources_between(df, main_list):
    
    df['num_sources_between'] = 'X'
    
    for list_tupel in main_list:
        instance_list = list_tupel[0]
        index_list = list_tupel[1]
        
        counter = 0
        for index in list(range(instance_list[0], instance_list[1])):
            if df.loc[index,'candidate_source'] == 'B':
                counter += 1
                
        df.loc[index_list,'num_sources_between'] = counter
        
    #return df
    
def run_dist_feats(list_of_tuples,df):
    
    main_list = create_instance_list(list_of_tuples,df)
    find_sc_dist(df, main_list)
    find_num_conts_between(df, main_list)
    find_num_sources_between(df, main_list)
    
    return df

def main():
    filepath = './../../Data/parc_features/parc_dev_features.tsv'
    df = pd.read_csv(filepath, delimiter='\t', index_col=0)

    instance_output = instance_generation.collect_instances_main(df)

    gold_label_list = list()
    pair_list = list()
    for instance_list in instance_output:
        gold_label_list.append(instance_list[2])
        pair_tupel = (instance_list[0], instance_list[1])
        pair_list.append(pair_tupel)

    print(gold_label_list[:4])
    print(pair_list[:4])

    main_list = create_instance_list(pair_list, df)
    #find_sc_dist(df, main_list)
    #find_num_conts_between(df, main_list)
    #find_num_sources_between(df, main_list)

    print(df.head())

