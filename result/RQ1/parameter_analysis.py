import pandas as pd
import os
import numpy as np

folder = './full_dataset/'
datasets = [
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

def seqDist(seq1, seq2):
    assert len(seq1) == len(seq2)
    simTokens = 0
    numOfPar = 0
    branch_point = len(seq1)

    for token1, token2 in zip(seq1, seq2):
        if token1 == '<*>':
            numOfPar += 1
            continue
        if token1 == token2:
            simTokens += 1 

    retVal = float(simTokens) / len(seq1)

    return retVal, branch_point-simTokens

st_dict = {
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

lengths = []
pair_below_threshold = []
branch_points = []
total_templates = 0
for dataset in datasets:
    pair_below_threshold = []
    file_path = os.path.join(folder, f"{dataset}/{dataset}_full.log_templates.csv")
    df = pd.read_csv(file_path)
    templates_by_length = {}
    templates = df['EventTemplate']
    for template in templates:
        length = len(template.split())
        lengths.append(length)
        if length not in templates_by_length.keys():
            templates_by_length[length] = [template]
        else:
            templates_by_length[length].append(template)
            
    # Calculate the pairwise sequence distances
    for length in templates_by_length.keys():
        templates_list = templates_by_length[length]
        # Determine if the templates are paired
        paired_flag = [False] * len(templates_list)
        for i in range(len(templates_list)):
            # Only check if the template is not paired
            if paired_flag[i]: continue
            paired_list = []
            for j in range(i + 1, len(templates_list)):
                if paired_flag[j]: continue
                sim, branch_point = seqDist(templates_list[i].split(), templates_list[j].split())
                threshold = st_dict.get(length, st_dict['others'])
                # Say they're paired if the distance is higher than the threshold
                if sim >= threshold:
                    paired_flag[j] = True
                    paired_list.append(templates_list[j])
                    branch_points.append(branch_point)
            if paired_list != []:
                pair_below_threshold.append(len(paired_list))
            else:
                pair_below_threshold.append(1)
                
    print(np.percentile(pair_below_threshold, 99))

        
# Calculate the three quartiles
Q1 = np.percentile(lengths, 25)
Q2 = np.percentile(lengths, 50)
Q3 = np.percentile(lengths, 75)

print('length quartiles:', Q1, Q2, Q3)


    
