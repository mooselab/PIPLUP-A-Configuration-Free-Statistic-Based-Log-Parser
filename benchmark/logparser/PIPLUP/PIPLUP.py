import regex as re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
from logparser.utils.preprocessing import preprocess
from logparser.utils.postprocessing import correct_single_template
import time

path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        ' ', ',', '!', ';', ':',
        '=', '|', '"', "'", '+', '.',
        '[', ']', '(', ')', '{', '}'
    }


class Logcluster:
    def __init__(self, logTemplate='', logIDL=None, br_thresh=2):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL
        self.pathList = [logTemplate]
        self.templateList = [correct_single_template(' '.join(logTemplate))]
        self.br_thresh = br_thresh
        self.hit_time = 1

    # Check whether template1 can describe template2 (template2 can be merged into template1)
    def templateMatches(self, template1, template2):
        # Escape all parts of the template except the <*> placeholders
        parts = re.split(r"<\*>", template1)
        regex_pattern = ".*?".join(re.escape(part) for part in parts)
        
        # Add anchors to ensure the entire template2 string is matched
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.fullmatch(regex_pattern, template2, re.DOTALL))
    
    def seqSimilarity(self, seq1, seq2):
        sim = 0
        for t1, t2 in zip(seq1, seq2):
            if t1 == t2 or '<*>' in t1: sim += 1
        return sim/len(seq1)
    
    def updateSequence(self, seq):
        # Has new token? 
        for path in self.pathList: 
            new_token = False
            for pos, token in enumerate(seq):
                if token != path[pos] and '<*>' not in self.pathList[0][pos]:
                    new_token = True
                    break
            if not new_token: return False

        # Is the path list full?
        if len(self.pathList) < self.br_thresh:
            # Add to the path list
            self.pathList.append(seq)
        # If it is full, merge with the closest path
        else:
            maxSim, target = 0, None
            for t in self.pathList:
                if self.seqSimilarity(seq, t) > maxSim:
                    maxSim = self.seqSimilarity(t, seq)
                    target = t
            # Start merging
            for pos, (t1, t2) in enumerate(zip(target, seq)):
                if t1 != t2:
                    # Collapse into a variable
                    # Protect the head and tail for readability
                    var = '<*>'
                    head, tail = '', ''
                    min_length = min(len(t1), len(t2))
                    for i in range(min_length):
                        char = t2[i]
                        if char not in path_delimiters: break
                        flag = True
                        if t1[i] != char: 
                            flag = False
                            break
                        if flag: head += char
                        else: break
                    for i in range(min_length):
                        char = t2[-i-1]
                        if char not in path_delimiters: break
                        flag = True
                        if t1[-i-1] != char: 
                            flag = False
                            break
                        if flag: tail = char + tail
                        else: break
                    var = head + var + tail
                    
                    # Update all lists
                    for p in self.pathList: p[pos] = var
        
        # Clean the list 
        self.pathList = np.unique(self.pathList, axis=0).tolist()
        # Update the template list
        self.templateList = [correct_single_template(' '.join(t)) for t in self.pathList]
        # Clean the templates
        newList = [True for _ in self.templateList]
        for i, t1 in enumerate(self.templateList):
            if newList[i] == False: continue
            for j in range(i+1, len(self.templateList)):
                t2 = self.templateList[j]
                if self.templateMatches(t2, t1): newList[i] = False; break
                if self.templateMatches(t1, t2): newList[j] = False
        self.templateList = [item for i, item in enumerate(self.templateList) if newList[i]]
        
        return True



class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', 
                 br_thresh=2, update_population=385, sim_thresh='default', keep_para=False):

        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        if sim_thresh == 'default':
            self.st_dict = {
                1: 1,
                2: (2-1)/2,
                3: (3-1)/3,
                4: (4-1)/4,
                5: (5-2)/5, 
                6: (6-2)/6, 
                7: (7-3)/7,
                8: (8-3)/8,
                9: (9-3)/9,
                'others': 0.5
            }
        else:
            try:
                self.st_dict = {'others': float(sim_thresh)}
            except:
                raise ValueError("sim_thresh must be 'default' or a float value.")
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.update_population = update_population
        self.log_format = log_format
        self.keep_para = keep_para
        self.commonword_list = None
        self.storage_tree = {'None':{}}
        self.shallow_tree = {'None':[]}
        self.br_thresh = br_thresh

    def treeSearch(self, seq):
        retLogClust = None
        
        # Find the frequent word
        constant = 'None'
        for token in self.shallow_tree.keys():
            if token in seq: 
                constant = token
                break
        
        # Search on the second layer
        seqLen = len(seq)
        
        if seqLen not in self.storage_tree[constant].keys():
            return retLogClust, constant, seq
        else:
            logClustL = self.storage_tree[constant][seqLen]
        
        retLogClust, new_template = self.fastMatch(logClustL, seq)

        return retLogClust, constant, new_template

    def addSeqToPrefixTree(self, logClust):
        seqLen = len(logClust.logTemplate)
        # Default first token is None
        constant = 'None'
        
        for token in self.shallow_tree.keys(): 
            if token in logClust.logTemplate: 
                constant = token
                break

        if seqLen not in self.storage_tree[constant].keys():
            self.storage_tree[constant][seqLen] = [logClust]
        else:
            self.storage_tree[constant][seqLen].append(logClust)

        self.shallow_tree[constant].append(logClust)
        
        return constant

    #seq1 is template
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0
        for token1, token2 in zip(seq1, seq2):
            if '<*>' in token1:
                if not any(ch.isalnum() for ch in token1): 
                    numOfPar += 1
                    continue
            if token1 == token2:
                simTokens += 1 

        retVal = float(simTokens) / len(seq1)
        return retVal, numOfPar


    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            # Directly take the same one
            if logClust.logTemplate == seq:
                maxSim = 1
                maxClust = logClust
                break
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim>maxSim or (curSim==maxSim and curNumOfPara>maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if len(seq) in self.st_dict.keys(): st = self.st_dict[len(seq)]
        else: st = self.st_dict['others']

        new_template = seq
        if maxSim >= st:
            retLogClust = maxClust  
            new_template = self.getTemplate(seq, retLogClust.logTemplate)
            
        return retLogClust, new_template

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []
        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')
            i += 1
        return retVal
    
    # Check whether template1 can describe template2 (template2 can be merged into template1)
    def templateMatches(self, template1, template2):
        # Escape all parts of the template except the <*> placeholders
        parts = re.split(r"<\*>", template1)
        regex_pattern = ".*?".join(re.escape(part) for part in parts)
        
        # Add anchors to ensure the entire template2 string is matched
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.fullmatch(regex_pattern, template2, re.DOTALL))
    
    
    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]

        df_event = {c:[] for c in ['EventId', 'EventTemplate', 'Occurrences']}
        # Use the templates
        for i, logClust in enumerate(logClustL):
            templates = logClust.templateList
            
            if len(templates) == 1:
                template_id = hashlib.md5(templates[0].encode('utf-8')).hexdigest()[0:8]
                occurrence = len(logClust.logIDL)
                for logID in logClust.logIDL:
                    
                    logID -= 1
                    log_templates[logID] = templates[0]
                    log_templateids[logID] = template_id
                if templates[0] in df_event['EventTemplate']:
                    idx = df_event['EventTemplate'].index(templates[0])
                    df_event['Occurrences'][idx] += occurrence
                else:
                    df_event['EventTemplate'].append(templates[0])
                    df_event['EventId'].append(template_id)
                    df_event['Occurrences'].append(occurrence)
            else:
                template_patterns, template_ids = [], []
                for template in templates:
                    pattern = re.escape(template).replace(re.escape("<*>"), ".*?")
                    pattern = f"^{pattern}$"
                    template_patterns.append(pattern)
                    template_ids.append(hashlib.md5(template.encode('utf-8')).hexdigest()[0:8])
                # Get the lines and find their matchs
                # print(len(logClust.logIDL), print(template_patterns))
                for logID in logClust.logIDL:
                    logID -= 1
                    for j, pattern in enumerate(template_patterns):
                        if bool(re.fullmatch(pattern, self.df_log['Content'][logID], re.DOTALL)):
                            log_templates[logID] = templates[j]
                            log_templateids[logID] = template_ids[j]
                            if templates[j] in df_event['EventTemplate']:
                                idx = df_event['EventTemplate'].index(templates[j])
                                df_event['Occurrences'][idx] += 1
                            else:
                                df_event['EventTemplate'].append(templates[j])
                                df_event['EventId'].append(template_ids[j])
                                df_event['Occurrences'].append(1)
                            break

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1) 
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])

    def jaccardSimilarity(self, template1, template2):
        set1 = set([item for item in template1.split() if any(ch.isalnum() for ch in item)])
        set2 = set([item for item in template2.split() if any(ch.isalnum() for ch in item)])

        if len(set1.union(set2)) == 0: return 0
        return len(set1.intersection(set2))/len(set1.union(set2))

    def mergeTemplates(self, cluster, constant, oldTemplateList=[]):
        pattern = r"[a-zA-Z0-9]"
        # Find the cluster in the shallow tree
        cluster_list = self.shallow_tree[constant]
        # Iterate through the clusters with the same frequent word
        for i, t1 in enumerate(cluster.templateList):
            # Skip if it is not updated
            if t1 in oldTemplateList: continue
            # Skip if it is too general
            # Iterate through the clusters
            for logClust in cluster_list:
                # Skip if it is too general
                for j, t2 in enumerate(logClust.templateList):
                    if self.jaccardSimilarity(t1, t2) < 0.6: continue
                    # t1 described by t2
                    if len(t1.split()) >= len(t2.split()):
                        if self.templateMatches(t2, t1): 
                            cluster.templateList[i] = t2
                            break
                    # t2 described by t1
                    if len(t1.split()) <= len(t2.split()):
                        if self.templateMatches(t1, t2): 
                            logClust.templateList[j] = t1
    
    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        logCluL = []
        self.load_data()
        # self.get_frequent_words_from_log()
        count = 0
        matched_types = []
        use_sequence = []
        for idx, line in self.df_log.iterrows():
            logID = line['LineId']
            # Add preprocess
            if idx < 2000:
                logmessageL, sequence, matched = preprocess(line['Content'], estimation_stage=True)
                # print(matched_types)
                logmessageL = logmessageL.strip().split()
                matched_types.extend(matched)
                use_sequence = sequence
            else:
                if idx == 2000:
                    matched_types = set(matched_types)
                    # print(matched_types)
                    use_sequence = [i for i in use_sequence if i in matched_types]
                logmessageL = preprocess(line['Content'], estimation_stage=False, use_sequence=use_sequence).strip().split()
            
            matchCluster, first_constant, newTemplate = self.treeSearch(logmessageL)
            
            #Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID], br_thresh=self.br_thresh)
                logCluL.append(newCluster)
                constant = self.addSeqToPrefixTree(newCluster)
                self.mergeTemplates(newCluster, constant)
            
            #Add the new log message to the existing cluster
            else:
                # Increase the hit time
                matchCluster.hit_time += 1
                oldTemplate = matchCluster.logTemplate
                matchCluster.logTemplate = newTemplate
                constant = first_constant
                # The template changed, update the list
                oldTemplateList = matchCluster.templateList.copy()
                matchCluster.logIDL.append(logID)
                updated = matchCluster.updateSequence(logmessageL)
                
                if (oldTemplate != newTemplate or matchCluster.hit_time == self.update_population) and first_constant == 'None': 
                    # Find the first constant
                    for token in newTemplate:
                        if '<*>' not in token:
                            constant = token
                            break
                    
                    # Update the tree
                    # 1. Initialize the clusters
                    if constant in self.storage_tree.keys():
                        if len(logmessageL) not in self.storage_tree[constant].keys():
                            self.storage_tree[constant][len(logmessageL)] = []
                    else:
                        self.storage_tree[constant] = {len(logmessageL): []}
                        self.shallow_tree[constant] = []
                    
                    # 2. remove the cluster from 'None' in both trees and update the other clusters
                    if constant != 'None':
                        # Find the clusters with the new constant
                        shallowTreeCopy = self.shallow_tree['None'].copy()
                        for cluster in self.shallow_tree['None']:
                            if constant in cluster.logTemplate: 
                                self.shallow_tree[constant].append(cluster)
                                if len(cluster.logTemplate) in self.storage_tree[constant].keys():
                                    self.storage_tree[constant][len(cluster.logTemplate)].append(cluster)
                                else:
                                    self.storage_tree[constant][len(cluster.logTemplate)] = [cluster]
                                    
                                index = self.storage_tree['None'][len(cluster.logTemplate)].index(cluster)
                                self.storage_tree['None'][len(cluster.logTemplate)].pop(index)
                                index = shallowTreeCopy.index(cluster)
                                shallowTreeCopy.pop(index)
                        self.shallow_tree['None'] = shallowTreeCopy
                    
                if updated:
                    self.mergeTemplates(matchCluster, constant, oldTemplateList)


            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))


        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(logCluL)

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)
    

    def log_to_dataframe(self, log_file, regex, headers, logformat):
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


    def generate_logformat_regex(self, logformat):
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

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list
