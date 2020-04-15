def extract_cue_labels(attribution_df):
    '''
    Takes an attribution pandas df and returns the same df with an added column "cue_label"
    
    :param attribution_df: a pandas dataframe with conll attribution column format
    
    :returns new_df: the same df with an added column "cue_label": a binary label for whether each token is a cue
    '''
    new_df = attribution_df.copy()
    cue_info = [0]*len(new_df["word"])
    
    # First, gather spans of all cues
    attribution_info = attribution_df["attribution"]
    num_cues = len(attribution_info[0].split(" "))
    cues = [] 
    attribution_part = "CUE"
    for i in range(num_cues):
        cues.append([])
    for att_index in range(num_cues):
        start_index = None
        for i in range(len(attribution_info)):
            attribution_string = attribution_info[i].split(" ")[att_index].split("-")
            if attribution_string != ["_"] and attribution_string[2] == "NE":
                # Skip nested attributions
                continue
            if attribution_string == ["_"] or attribution_string[1] != attribution_part:
                # Not inside a span
                if start_index != None:
                    # Found the end of a span we were in
                    cues[att_index].append((start_index, i))
                    start_index = None
            else:
                IOB_label, att_part = attribution_string[0], attribution_string[1]
                if att_part == attribution_part and IOB_label == "B":
                    start_index = i
                    
    for span_list in cues:
        last_index_in_cue = -1
        for span in span_list:
            if span[-1] - 1 > last_index_in_cue:
                last_index_in_cue = span[-1]-1
        if last_index_in_cue != -1:
            cue_info[last_index_in_cue] = 1
    new_df["cue_label"] = cue_info
    return new_df