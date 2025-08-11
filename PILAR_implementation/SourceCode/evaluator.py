"""
Description : This file implements the function to evaluation accuracy of log parsing
Author      : LogPAI team
License     : MIT
"""

import sys
import pandas as pd
from collections import defaultdict
try:
    from scipy.misc import comb
except ImportError as e:
    from scipy.special import comb
from tqdm import tqdm

def evaluate_agreement(file_1, file_2):
    df_file1 = pd.read_csv(file_1)
    df_file2 = pd.read_csv(file_2)

    eventlist_1 = df_file1['EventTemplate']
    eventlist_2 = df_file2['EventTemplate']

    count = 0
    for index in range(0,4999,1):
        line_1 = eventlist_1[index+5000]
        line_2 = eventlist_2[index]
        if(line_1.__eq__(line_2)):
            count = count+1

    agreement = count/5000
    return agreement

def evaluate_sample(groundtruth, parsedresult):
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, quoting=3, nrows=2000)
    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    # df_parsedlog = df_parsedlog.loc[null_logids]
    (precision, recall, f_measure, accuracy) = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])
    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f' % (
    precision, recall, f_measure, accuracy))
    return f_measure, accuracy

def evaluate(df_groundtruth, df_parsedlog, filter_templates=None):
    """ Evaluation function to org_benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """ 
    # df_groundtruth = pd.read_csv(groundtruth)
    # df_parsedlog = pd.read_csv(parsedresult)
    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    (GA, FGA) = get_accuracy(df_groundtruth['EventTemplate'], df_parsedlog['EventTemplate'], filter_templates)
    print('Grouping_Accuracy (GA): %.4f, FGA: %.4f,'%(GA, FGA))
    return GA, FGA


def get_accuracy(series_groundtruth, series_parsedlog, filter_templates=None):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('groundtruth')
    accurate_events = 0 # determine how many lines are correctly parsed
    accurate_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    # for ground_truthId in tqdm(series_groundtruth_valuecounts.index):
    for ground_truthId, group in tqdm(grouped_df):
        # logIds = series_groundtruth[series_groundtruth == ground_truthId].index
        series_parsedlog_logId_valuecounts = group['parsedlog'].value_counts()
        if filter_templates is not None and ground_truthId in filter_templates:
            for parsed_eventId in series_parsedlog_logId_valuecounts.index:
                filter_identify_templates.add(parsed_eventId)
        if series_parsedlog_logId_valuecounts.size == 1:
            parsed_eventId = series_parsedlog_logId_valuecounts.index[0]
            if len(group) == series_parsedlog[series_parsedlog == parsed_eventId].size:
                if (filter_templates is None) or (ground_truthId in filter_templates):
                    accurate_events += len(group)
                    accurate_templates += 1
    # print("filter templates: ", len(filter_templates))
    # print("total messages: ", len(series_groundtruth[series_groundtruth.isin(filter_templates)]))
    # print("set of total: ", len(filter_identify_templates))
    # print(accurate_events, accurate_templates)
    if filter_templates is not None:
        GA = float(accurate_events) / len(series_groundtruth[series_groundtruth.isin(filter_templates)])
        PGA = float(accurate_templates) / len(filter_identify_templates)
        RGA = float(accurate_templates) / len(filter_templates)
    else:
        GA = float(accurate_events) / len(series_groundtruth)
        PGA = float(accurate_templates) / len(series_parsedlog_valuecounts)
        RGA = float(accurate_templates) / len(series_groundtruth_valuecounts)
    # print(FGA, RGA)
    FGA = 0.0
    if PGA != 0 or RGA != 0:
        FGA = 2 * (PGA * RGA) / (PGA + RGA)
    return GA, FGA



