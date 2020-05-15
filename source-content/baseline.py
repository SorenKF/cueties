from instance_generation import content_in_sentence, candidate_sources, collect_span_dict, check_for_contents, collect_candidate_sources, collect_instances, compare_spans, collect_instances_main
import pandas as pd
import time
from collections import defaultdict

#Choose your input path
input_path = "C:/Users/Stell/Documents/Attribution_System/content_features_output/polnear_features/polnear_train_features.tsv"
#input_path = "C:/Users/Stell/Documents/Attribution_System/content_features_output/parc_features/parc_train_features.tsv"
df = pd.read_csv(input_path, sep='\t', header=0, index_col=0)

#create new columns
df["content_in_sentence"] = content_in_sentence(df)
df["candidate_source_label"] = candidate_sources(df)

#label B's of sources and B's of contents that should be linked together with the same number
print("numbering links...")
df["content_number"] = 0
df["source_number"] = 0
counter = 0

for index, cont_label in enumerate(df['content_label_gold']):
    if df.iloc[index]['content_label_gold'] == 'B':
        counter += 1
        if df.iloc[index-1]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index-1, 'source_number'] = counter
        elif df.iloc[index-2]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index-2, 'source_number'] = counter
        elif df.iloc[index-3]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index-3, 'source_number'] = counter
        elif df.iloc[index-4]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index-4, 'source_number'] = counter
        elif df.iloc[index-5]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index-5, 'source_number'] = counter
        elif df.iloc[index+1]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index+1, 'source_number'] = counter
        elif df.iloc[index+2]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index+2, 'source_number'] = counter
        elif df.iloc[index+3]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index+3, 'source_number'] = counter            
        elif df.iloc[index+4]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index+4, 'source_number'] = counter
        elif df.iloc[index+5]['candidate_source_label'] == 'B':
            df.at[index, 'content_number'] = counter
            df.at[index+5, 'source_number'] = counter 

# label the complete spans with that number
print("numbering spans...")
for index, content_label in enumerate(df["candidate_source_label"]):
    if df.iloc[index]["candidate_source_label"] == 'I' and df.iloc[index-1]["source_number"] != 0:
        df.at[index, "source_number"] = df.at[index-1, "source_number"]            
            
for index, content_label in enumerate(df["content_label_gold"]):
    if df.iloc[index]["content_label_gold"] == 'I' and df.iloc[index-1]["content_number"] != 0:
        df.at[index, "content_number"] = df.at[index-1, "content_number"]
     
    
#write the new df to a csv to check what on earth I am actually doing
df.to_csv("C:/Users/Stell/Documents/Attribution_System/source-content/newest_df_polnear.csv", sep='\t', encoding='utf-8')
#intermittant saved df: df = pd.read_csv("C:/Users/Stell/Documents/Attribution_System/source-content/newest_df_parc.csv", sep='\t', header=0, index_col=0)


# get indices from the instances created by Nathan without the labels
print("labelling instances...")
result = collect_instances_main(df)
new_list = []
for source, content, label in result:
    new_list.append([source, content])
return_list = []

#label instances with predictions from the baseline
for source, content in new_list:
    x, y = source
    k, l = content
    if df.iloc[x]["source_number"] == df.iloc[k]["content_number"] and df.iloc[y-1]["source_number"] == df.iloc[l-1]["content_number"] and df.iloc[x]["source_number"] != 0:
        return_list.append([source, content, 1])
    else:
        return_list.append([source, content, 0])


#evaluation on token-level
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

y_true = []
for source, content, label in result:
    y_true.append(label)
y_pred = []
for source, content, prediction in return_list:
    y_pred.append(prediction)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print('precision: ', precision, 'recall: ', recall, 'f1: ', f1, 'accuracy: ', accuracy)