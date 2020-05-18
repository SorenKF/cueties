import pandas as pd
from instance_generation import content_in_sentence, candidate_sources, collect_span_dict, check_for_contents, collect_candidate_sources, collect_instances, compare_spans, collect_instances_main
from collections import defaultdict

df = pd.read_csv("newest_df_parc.csv", sep='\t', header=0, index_col=0)
#df = pd.read_csv("newest_df_polnear.csv", sep='\t', header=0, index_col=0)

# get indices from the instances created by Nathan without the labels
def main():
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
            
    return(return_list)

if __name__ == "__main__":
    main()

# y_true = []
# for source, content, label in result:
#     y_true.append(label)
# y_pred = []
# for source, content, prediction in return_list:
#     y_pred.append(prediction)
    
####################    What you need is return_list!!!