import pandas as pd
import os
import numpy as np

folders = ['different_br_thresh/', 'different_sim_thresh/']
datasets = ['Proxifier','Linux','Apache','Zookeeper','Mac','Hadoop','OpenStack','HealthApp','HPC','OpenSSH','Average']

for folder in folders:
    files = os.listdir('./result/RQ1/'+folder)
    print('Analyzing results from folder: %s'%folder)
    results = {dataset:{'GA': [], 'PA': [], 'FGA': [], 'FTA': []} for dataset in datasets}

    for f in files:
        # Collect the four metrics
        df = pd.read_csv('./result/RQ1/'+folder+f, header=1)
        for dataset in datasets:
            results[dataset]['GA'].append(df[dataset][3])
            results[dataset]['PA'].append(df[dataset][4])
            results[dataset]['FGA'].append(df[dataset][5])
            results[dataset]['FTA'].append(df[dataset][8])

    # Get the variances
    for dataset in datasets:
        for metric in ['GA', 'PA', 'FGA', 'FTA']:
            results[dataset][metric] = '%.3f/%.3f'%(min(results[dataset][metric]), max(results[dataset][metric]))
            # results[dataset][metric] = '%.3f'%((max(results[dataset][metric])-min(results[dataset][metric]))/max(results[dataset][metric])*100)
    print(results)
    
