from DictionarySetUp import GramDict
from Parser import Parser
from evaluator import evaluate
from evaluator import evaluate_sample
from evaluator import evaluate_agreement
from DictSummary import DictEvaluate
from DictSummary import DictPrint
from DictionarySetUp import EntropyOnline

from scipy import stats
import numpy as np
import statistics
from Common import cohend
from Common import cliffsDelta
import re
import os
import time
import csv

import pandas as pd
from utils.template_level_analysis import evaluate_template_level
from utils.PA_calculator import calculate_parsing_accuracy


eventoutput = 'Output/event.txt'
templateoutput = 'Output/template.csv'
separator = ' '
regex = [
        r'([\w-]+\.)+[\w-]+(:\d+)', #url
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
]

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'regex': [],
        'st': 0.5,
        'depth': 4
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4        
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'st': 0.5,
        'depth': 4
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'st': 0.7,
        'depth': 5      
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'st': 0.39,
        'depth': 6        
        },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'st': 0.2,
        'depth': 6   
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'st': 0.2,
        'depth': 4
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'st': 0.6,
        'depth': 3
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'depth': 5   
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'st': 0.5,
        'depth': 5
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.7,
        'depth': 6   
        },
}

HDFS_file = 'HDFS.log'
Hadoop_file = 'Hadoop.log'
Spark_file = 'Spark.log'
Zookeeper_file = 'Zookeeper.log'
BGL_file = 'BGL.log'
HPC_file = 'HPC.log'
Thunderbird_file = 'Thunderbird.log'
Windows_file = 'Windows.log'
Linux_file = 'Linux.log'
Android_file = 'Android.log'
Apache_file = 'Apache.log'
OpenSSH_file = 'SSH.log'
OpenStack_file = 'OpenStack.log'
Mac_file = 'Mac.log'
HealthApp_file = 'HealthApp.log'
Proxifier_file = 'Proxifier.log'

HDFS_num = 10
Hadoop_num = 10
Spark_num = 10
Zookeeper_num = 10
BGL_num = 10
HPC_num = 10
Thunderbird_num = 10
Windows_num = 10
Linux_num = 5
Android_num = 10
OpenSSH_num = 10
OpenStack_num = 6
Mac_num = 9
HealthApp_num = 10
Apache_num = 5
#Proxifier_num = 10

directory = 'Sampledata/'

def generate_logformat_regex(logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

def log_to_dataframe(log_file, regex, headers, logformat):
    """ Function to transform log file to dataframe 
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    print("Total lines: ", len(logdf))
    return logdf

datasets_full = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]


agreement_result = []
with open(os.path.join('./result', 'summary_[otc=0,complex=0,frequent=0].csv'), 'w') as csv_file:
        fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # fw.writerow(['Dataset', 'GA_time', 'PA_time', 'TA_time', 'parse_time', 'identified_templates',
        #              'ground_templates', 'GA', 'PA', 'FTA', 'PTA', 'RTA', 'OG', 'UG', 'MX'])
        fw.writerow(['Dataset', 'parse_time', 'identified_templates',
                'ground_templates', 'GA', 'PA', 'FGA', 'PTA', 'RTA', 'FTA'])
for dataset in datasets_full:
        print('Parsing dataset: ', dataset)
        t1 = time.time()
        logfile = '../PIPLUP/full_dataset/%s/%s_full.log'%(dataset, dataset)
        headers, regex = generate_logformat_regex(benchmark_settings[dataset]['log_format'])
        df_log = log_to_dataframe(logfile, regex, headers, benchmark_settings[dataset]['log_format'])
        
        gramdict_1 = GramDict(logfile, separator, benchmark_settings[dataset]['log_format'], benchmark_settings[dataset]['regex'], 1)

        tokenslist_1, singledict_1, doubledict_1, tridict_1 = gramdict_1.DictionarySetUp()
        ratio = 0.2
        '''
        
        while ratio <= 0.4:'''
        parser_1 = Parser(tokenslist_1, singledict_1, doubledict_1, tridict_1, ratio, dataset)

        parser_1.Parse(df_log)
        t2 = time.time()
        parse_time = t2-t1

        # Evaluation
        groundtruth = '../PIPLUP/full_dataset/%s/%s_full.log_structured.csv'%(dataset, dataset)
        parsedresult = './result/%s/%s_structured.csv'%(dataset, dataset)

        parsedresult = pd.read_csv(parsedresult, dtype=str)
        parsedresult.fillna("", inplace=True)
        groundtruth = pd.read_csv(groundtruth, dtype=str)
        
        
        print("Start compute grouping accuracy")
        # calculate grouping accuracy
        start_time = time.time()
        GA, FGA = evaluate(groundtruth, parsedresult)

        GA_end_time = time.time() - start_time
        print('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(GA_end_time))

        # calculate parsing accuracy
        start_time = time.time()
        PA = calculate_parsing_accuracy(groundtruth, parsedresult, None)
        PA_end_time = time.time() - start_time
        print('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(PA_end_time))

        # calculate template-level accuracy
        start_time = time.time()
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(dataset, groundtruth, parsedresult, None)
        TA_end_time = time.time() - start_time
        print('Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))

        result = dataset + ',' + \
                "{:.2f}".format(parse_time) + ',' + \
                str(tool_templates) + ',' + \
                str(ground_templates) + ',' + \
                "{:.3f}".format(GA) + ',' + \
                "{:.3f}".format(PA) + ',' + \
                "{:.3f}".format(FGA) + ',' + \
                "{:.3f}".format(PTA) + ',' + \
                "{:.3f}".format(RTA) + ',' + \
                "{:.3f}".format(FTA) + '\n'

        with open(os.path.join('./result', 'summary_[otc=0,complex=0,frequent=0].csv'), 'a') as summary_file:
                summary_file.write(result)
        '''exit()
        agreement = evaluate_agreement("Output/event1.txt","Output/event2.txt")

        agreement_result.append([logfile_1, logfile_2, ratio, agreement])'''

'''df_result = pd.DataFrame(agreement_result, columns=['File1', 'File2', 'Para', 'Agreement'])
df_result.to_csv('Logent_agreement_BGL.csv')'''



# directory = 'Sampledata/'
#
# for index in range(0,OpenStack_num,1):
#         logfile = directory + OpenStack_file + '.part' + str(index)
#         print(logfile)
#         gramdict = GramDict(logfile, separator, OpenStack_format, OpenStack_Regex, 1)
#         tokenslist, singledict, doubledict, tridict = gramdict.DictionarySetUp()
#
#         ratio = 0.1
#         while ratio < 0.4:
#                 print(ratio)
#                 parser = Parser(tokenslist, singledict, doubledict, tridict, ratio)
#                 parser.Parse()
#                 ratio = ratio + 0.01
#                 evaluate_sample('GroundTruth/OpenStack_2k.log_structured.csv', 'Output/event.txt')


# gramdict = GramDict(logfile, separator, Spark_format, Spark_Regex, 1)
# # tokenslist, singledict, doubledict, tridict, fourdict = gramdict.DictionarySetUp()
# tokenslist, singledict, doubledict, tridict= gramdict.DictionarySetUp()
#
# ratio = 0.1
# while ratio < 0.4:
#         print(ratio)
#         parser = Parser(tokenslist, singledict, doubledict, tridict, ratio)
#         parser.Parse()
#         ratio = ratio + 0.01
#         evaluate('GroundTruth/Spark_2k.log_structured.csv', 'Output/event.txt')

# parser = Parser(tokenslist, singledict, doubledict, tridict, 0.3)
# parser.Parse()

# onlineparser = EntropyOnline(logfile, separator, Andriod_format, Andriod_Regex, 0.1)
# onlineparser.Parse()
#
# OnlineEvents = open('Output/OLevent.txt').readlines()
# OfflineEvents = open('Output/event.txt').readlines()
#
# index = 0
# num = 0
# for OLevent in OnlineEvents:
#     OFFevent = OfflineEvents[index]
#     if OLevent == OFFevent:
#         num = num + 1
#     index = index + 1
#
# ratio = num/(index + 1)
# print(ratio)

# for s in steps:
#     print(s)
#     parser = Parser(tokenslist, singledict, doubledict, tridict, s)
#     dList, sList = parser.ParseTest()
#     print("pvalue: " + str(stats.ttest_ind(dList, sList)))
#     print("cohend: " + str(cohend(sList, dList)))
#     print("mean: " + str(statistics.mean(sList) - statistics.mean(dList)))
#     print("cliff: " + str(cliffsDelta(sList,dList)))

#DictPrint(entropydict)
# evaluate('GroundTruth/OpenStack_2k.log_structured.csv', 'Output/event.txt')