import pandas as pd
import numpy as np
import re

def run_model(paths, niter):
    regex = re.compile('[^a-zA-Z> ]')
    paths.rename(columns={paths.columns[0]: "Paths"}, inplace=True)
    paths['Paths'] = paths['Paths'].apply(lambda x: regex.sub('', x))
    markov_conversions = first_order(paths, niter)
    return markov_conversions

def first_order(paths, niter):
    paths = np.array(paths).tolist()
    sublist = []
    for path in paths:
        for touchpoint in path:
            userpath = touchpoint.split(' > ')
            sublist.append(userpath)
    paths = sublist
    unique_touch_list = np.unique([item for sublist in paths for item in sublist])

    # get total last touch conversion counts
    conv_dict = {}
    total_conversions = 0
    for item in unique_touch_list:
        conv_dict[item] = 0
    for path in paths:
        if 'conv' in path:
            total_conversions += 1
            conv_dict[path[-2]] += 1

    numSimulations = niter
    transitionStates = {}

    for x in unique_touch_list:
        for y in unique_touch_list:
            transitionStates[x + ">" + y] = 0

    for possible_state in unique_touch_list:
        if possible_state != "null" and possible_state != "conv":
            # print(possible_state)
            for user_path in paths:

                if possible_state in user_path:
                    indices = [i for i, s in enumerate(user_path) if possible_state in s]
                    for col in indices:
                        transitionStates[user_path[col] + ">" + user_path[col + 1]] += 1

    transitionMatrix = []
    actual_paths = []
    for state in unique_touch_list:

        if state != "null" and state != "conv":
            counter = 0
            index = [i for i, s in enumerate(transitionStates) if state + '>' in s]
            for col in index:
                if transitionStates[list(transitionStates)[col]] > 0:
                    counter += transitionStates[list(transitionStates)[col]]
            for col in index:
                if transitionStates[list(transitionStates)[col]] > 0:
                    state_prob = float((transitionStates[list(transitionStates)[col]])) / float(counter)
                    actual_paths.append({list(transitionStates)[col]: state_prob})
    transitionMatrix.append(actual_paths)

    flattened_matrix = [item for sublist in transitionMatrix for item in sublist]
    transState = []
    transMatrix = []
    for item in flattened_matrix:
        for key in item:
            transState.append(key)
        for key in item:
            transMatrix.append(item[key])

    df = pd.DataFrame({'paths': transState,
                       'prob': transMatrix})
    conv_sim_array = []

    for iterations in range(1, numSimulations):
        activityList = ["start"]
        firstChannel = np.random.choice(df[df['paths'].str.contains("start>")]['paths'].tolist(), replace=True,
                                        p=df[df['paths'].str.contains("start>")]['prob'].tolist())
        activityNow = firstChannel[firstChannel.index(">") + 1:]
        activityList.append(activityNow)

        while activityNow != 'conv' and activityNow != 'null':
            change = np.random.choice(df[df['paths'].str.contains(activityNow + '>')]['paths'].tolist(), replace=True,
                                      p=df[df['paths'].str.contains(activityNow + '>')]['prob'].tolist())
            activityNow = change[change.index(">") + 1:]
            activityList.append(activityNow)
        conv_sim_array.append(activityList)

    count = 0
    for smaller_list in conv_sim_array:
        if (smaller_list[-1] == "conv"):
            count += 1
    initial_cvr = float(count) / float(numSimulations)

    # channel/tactic removal transition matrix
    for row in unique_touch_list:
        if row != 'conv' and row != 'null' and row != 'start':
            # print(row)
            df['minus_' + row] = np.where(df['paths'].str.contains(row + '>'), 0, df['prob'])

    removal_cvr_dict = {}
    print(unique_touch_list)
    for row in unique_touch_list:
        if row != 'conv' and row != 'null' and row != 'start':
            print("Running simulations for removal of " + row)

            conv_sim_array = []
            # print("Starting new path sim")
            for iterations in range(1, numSimulations):
                activityList = ["start"]
                firstChannel = np.random.choice(df[df['paths'].str.contains("start>")]['paths'].tolist(), replace=True,
                                                p=df[df['paths'].str.contains("start>")]['prob'].tolist())
                activityNow = firstChannel[firstChannel.index(">") + 1:]
                activityList.append(activityNow)
                if activityNow == row:
                    activityList.append("null")
                else:
                    while activityNow != 'conv' and activityNow != 'null':
                        change = np.random.choice(df[df['paths'].str.contains(activityNow + '>')]['paths'].tolist(),
                                                  replace=True, p=df[df['paths'].str.contains(activityNow + '>')][
                                'minus_' + row].tolist())
                        if change[change.index(">") + 1:] == row:
                            activityList.append("null")
                            break
                        else:
                            activityNow = change[change.index(">") + 1:]
                            activityList.append(activityNow)
                conv_sim_array.append(activityList)

            #print(conv_sim_array)
            count = 0
            for smaller_list in conv_sim_array:
                if (smaller_list[-1] == "conv"):
                    count += 1
            cvr = float(count) / float(numSimulations)
            removal_cvr_dict[row] = cvr

    weighting_denominator = 0.0000
    removal_effect_dict = {}
    for k in removal_cvr_dict:
        removal_effect_dict[k] = removal_cvr_dict[k] / initial_cvr - 1
        weighting_denominator += removal_cvr_dict[k] / initial_cvr - 1
    #print("initial CVR: %f" % (initial_cvr))
    #print("Removal CVR Dict: %s" % (removal_cvr_dict))
    #print("Weighing denominator: %f " % (weighting_denominator))
    fractional_multiplier_dict = {}
    markov_attributed_conversions = {}
    for k in removal_effect_dict:
        #print(removal_effect_dict[k])
        fractional_multiplier_dict[k] = removal_effect_dict[k] / weighting_denominator
        markov_attributed_conversions[k] = round(fractional_multiplier_dict[k] * total_conversions)
    conv_dict.pop('conv', None)
    conv_dict.pop('null', None)
    conv_dict.pop('start', None)
    return {'markov_conversions': markov_attributed_conversions,
            'removal_cvr': removal_cvr_dict,
            'last_touch_conversions': conv_dict,
            'transition_matrix': df,
           'initial_cvr': initial_cvr}