import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def extract_features_and_labels_generalised(df, header='nerc', feature_selection='all'):
    """
    Adapted from Quirine's Machine Learning for NLP assignment.
    Takes pd.DataFrame of features and labels and outputs two lists. One list of dictionaries
    containing the features, and the other a list of nerc labels.

    :param df: pd.DataFrame containing features and labels.
    :param str header: string of header of column containing labels. Default is 'nerc'
    :param str/list feature_selection: string that says all to include all features in df. Or list
        of feature labels to be included. The names must match column names in df.

    :returns two lists.
    """
    feature_list = []
    label_list = []

    feature_labels = list(df)
    feature_labels.remove(header)

    # Get list of dictionaries of features. Feature labels are the keys, feature value is the value.
    # If no selection is made, include all headers from df (except for NERC label) to feature dict.
    if feature_selection == 'all':
        for i in range(df.shape[0]):
            feature_dict = dict()
            for feature_label in feature_labels:
                feature_value = df.at[i, feature_label]
                feature_dict[feature_label] = feature_value
            feature_list.append(feature_dict)
    # If a selection is made, only include those features to feature dict.
    else:
        for i in range(df.shape[0]):
            feature_dict = dict()
            for feature_label in feature_selection:
                feature_value = df.at[i, feature_label]
                feature_dict[feature_label] = feature_value
            feature_list.append(feature_dict)

    # Get list of nerc labels
    label_list = list(df[header])

    return feature_list, label_list

def train_pred_logreg(train_feats, train_labels, test_feats):
    # Get features ready for model

    vec = DictVectorizer()
    train_feats_vec = vec.fit_transform(train_feats)

    # Make instance of model
    logisticRegr = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    model = logisticRegr.fit(train_feats_vec, train_labels)

    predict_feats_vec = vec.transform(test_feats)

    predictions = model.predict(predict_feats_vec)
    return predictions

def main():
    #df_gold_labels = pd.read_csv('source-content_feature_data/parc_dev_pairs.tsv', sep='\t', header=0, index_col=0)
    
    df_train = pd.read_csv('C:/Users/Stell/Documents/Attribution_System/source-content/source-content_feature_data/polnear_train_pairs.tsv', sep='\t', header=0, index_col=0)
    df_test = pd.read_csv('C:/Users/Stell/Documents/Attribution_System/source-content/source-content_test_pred_pairs/polnear_test_pred_pairs.tsv', sep='\t', header=0, index_col=0)
    #df_test['gold_labels'] = df_gold_labels['gold_labels']

    # Extract feats/labels
    train_feats, train_labels = extract_features_and_labels_generalised(df_train, header='gold_labels', feature_selection=['source_content_boundaries', 'content_length', 'source_length', 'distance_c2docstart', 'distance_c2sentstart', 'source_in_content', 's/c_distance', 'num_conts_between'])
    test_feats, test_labels = extract_features_and_labels_generalised(df_test, header='gold_labels', feature_selection=['source_content_boundaries', 'content_length', 'source_length', 'distance_c2docstart', 'distance_c2sentstart', 'source_in_content', 's/c_distance', 'num_conts_between'])


    predictions = train_pred_logreg(train_feats, train_labels, test_feats)

       # Write to df
    df_test['maxent_pred'] = predictions
    df_test.to_csv("C:/Users/Stell/Documents/Attribution_System/source-content/source-content_test_pred_pairs/polnear_test_pred_pairs_with_predictions.tsv", sep='\t', encoding='utf-8')

    print('CLASSIFICATION REPORT')
    print(classification_report(df_test['gold_labels'], df_test['maxent_pred']))
        
if __name__ == "__main__":
    main()