import pandas as pd
import numpy as np
import re


def run_model(paths):
    regex = re.compile('[^a-zA-Z> ]')
    paths.rename(columns={paths.columns[0]: "Paths"}, inplace=True)
    paths['Paths'] = paths['Paths'].apply(lambda x: regex.sub('', x))
    markov_conversions = first_order(paths)
    return markov_conversions


def calculate_removals(df, base_cvr):
    removal_effect_list = dict()
    channels_to_remove = df.drop(['conv', 'null', 'start'], axis=1).columns
    for channel in channels_to_remove:
        removal_cvr_array = list()
        removal_channel = channel
        removal_df = df.drop(removal_channel, axis=1)
        removal_df = removal_df.drop(removal_channel, axis=0)
        for col in removal_df.columns:
            one = float(1)
            row_sum = np.sum(list(removal_df.loc[col]))
            null_percent = one - row_sum
            if null_percent == 0:
                continue
            else:
                removal_df.loc[col]['null'] = null_percent
        removal_df.loc['null']['null'] = 1.0
        R = removal_df[['null', 'conv']]
        R = R.drop(['null', 'conv'], axis=0)
        Q = removal_df.drop(['null', 'conv'], axis=1)
        Q = Q.drop(['null', 'conv'], axis=0)
        t = len(Q.columns)

        N = np.linalg.inv(np.identity(t) - np.asarray(Q))
        M = np.dot(N, np.asarray(R))
        removal_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
        removal_effect = 1 - removal_cvr / base_cvr
        removal_effect_list[channel] = removal_effect
    return removal_effect_list


def first_order(paths):
    paths = np.array(paths).tolist()
    sublist = []
    total_paths = 0
    for path in paths:
        for touchpoint in path:
            userpath = touchpoint.split(' > ')
            sublist.append(userpath)
        total_paths += 1
    paths = sublist
    unique_touch_list = set(x for element in paths for x in element)
    # get total last touch conversion counts
    conv_dict = {}
    total_conversions = 0
    for item in unique_touch_list:
        conv_dict[item] = 0
    for path in paths:
        if 'conv' in path:
            total_conversions += 1
            conv_dict[path[-2]] += 1

    transitionStates = {}
    base_cvr = total_conversions / total_paths
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

    tmatrix = pd.DataFrame({'paths': transState,
                            'prob': transMatrix})
    # unique_touch_list = model['unique_touch_list']
    tmatrix = tmatrix.join(tmatrix['paths'].str.split('>', expand=True).add_prefix('channel'))[
        ['channel0', 'channel1', 'prob']]
    column = list()
    for k, v in tmatrix.iterrows():
        if v['channel0'] in column:
            continue
        else:
            column.append(v['channel0'])
    test_df = pd.DataFrame()
    for col in unique_touch_list:
        test_df[col] = 0.00
        test_df.loc[col] = 0.00
    for k, v in tmatrix.iterrows():
        x = v['channel0']
        y = v['channel1']
        val = v['prob']
        test_df.loc[x][y] = val
    test_df.loc['conv']['conv'] = 1.0
    test_df.loc['null']['null'] = 1.0
    R = test_df[['null', 'conv']]
    R = R.drop(['null', 'conv'], axis=0)
    Q = test_df.drop(['null', 'conv'], axis=1)
    Q = Q.drop(['null', 'conv'], axis=0)
    O = pd.DataFrame()
    t = len(Q.columns)
    for col in range(0, t):
        O[col] = 0.00
    for col in range(0, len(R.columns)):
        O.loc[col] = 0.00
    N = np.linalg.inv(np.identity(t) - np.asarray(Q))
    M = np.dot(N, np.asarray(R))
    base_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
    removal_effects = calculate_removals(test_df, base_cvr)
    denominator = np.sum(list(removal_effects.values()))
    allocation_amount = list()
    for i in removal_effects.values():
        allocation_amount.append((i / denominator) * total_conversions)
    # print(allocation_amount)
    markov_conversions = dict()
    i = 0
    for channel in removal_effects.keys():
        markov_conversions[channel] = allocation_amount[i]
        i += 1
    conv_dict.pop('conv', None)
    conv_dict.pop('null', None)
    conv_dict.pop('start', None)

    return {'markov_conversions': markov_conversions,
            'last_touch_conversions': conv_dict,
            'removal_effects': removal_effects,
            'base_cvr': base_cvr,
            'transition_matrix': test_df,
            'absorption_matrix': M
           }

