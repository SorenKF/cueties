import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer

def get_candidate_cue_df(df):
    '''
    Takes a pandas dataframe with at least a column "candidate_cue" and returns a new dataframe with 
    only rows that have a 1 in the "candidate_cue" column
    
    :param df: a pandas DataFrame as specified above
    
    :returns new_df: the same df with only "candidate_cue" rows
    '''
    new_df = df.loc[df['candidate_cue'] == 1]
    return new_df

def get_feature_columns(df):
    '''
    Takes a pandas df and returns the same df but without 'dependency_head', 'ne_info', 'short_ne', 
    attribution', 'filename', 'candidate_cue', and index information
    
    :param df: a pandas DataFrame object, as outlined above
    
    :returns new_df: the same df with adjusted columns
    '''
    new_df = df.drop(columns = ['dependency_head', 'ne_info', 'attribution', 'ne_short', 
                                'doc_token_number', 'sentence_number', 'sentence_token_number',
                                'filename', 'candidate_cue'])
    return new_df

def train_and_test(train_df, test_df):
    '''
    Takes a training and a testing dataframe with feature columns feature and gold cue labels,
    trains a k-NN model, and outputs the results of the testing in the form of a list of predicted labels
    
    :param train_df, test_df: two pandas DFs with feature columns and a cold cue_label col
    
    :returns predicted_cue_labels: a list of predicted labels from trained model
    '''
    X_train = train_df.drop(columns="cue_label")
    Y_train = train_df["cue_label"].values
    
    train_features = X_train.to_dict("records")
    
    vec = DictVectorizer()
    
    train_vectorized = vec.fit_transform(train_features)
    
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(train_vectorized, Y_train)
    
    X_test = test_df.drop(columns="cue_label")
    Y_test = test_df["cue_label"]
    test_features = X_test.to_dict("records")
    test_vectorized = vec.transform(test_features)
    
    predicted_cue_labels = knn.predict(test_vectorized)
    
    return predicted_cue_labels
    
def main():
    
    start = time.time()
    
    train_file = "./Data/feature_files/parc_features/parc_train_features.tsv"
    test_file = "./Data/feature_files/parc_features/parc_test_features.tsv"
    
    train_df = pd.read_csv(train_file, sep='\t', index_col=0, header=0)
    test_df = pd.read_csv(test_file, sep='\t', index_col=0, header=0)
    
    CC_train_df = get_candidate_cue_df(train_df)
    train_df = get_feature_columns(CC_train_df)
    CC_test_df = get_candidate_cue_df(test_df)
    test_df = get_feature_columns(CC_test_df)
    
    predicted_cue_labels = train_and_test(train_df, test_df)
    
    CC_test_df["predicted_cue_label"] = predicted_cue_labels
    
    CC_test_df.to_csv("parc_test_results.tsv", sep="\t")
    
    print(f"Finished. Time elapsed: {time.time()-start}")
    
if __name__ == "__main__":
    main()
    