"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os

sys.path.append('../')

from logparser.PIPLUP import LogParser
import argparse
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average


datasets_rq1 = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Mac",
    "Hadoop",
    "OpenStack",
    "HealthApp",
    "HPC",
    "OpenSSH",
]


datasets_full = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Mac",
    "Hadoop",
    "OpenStack",
    "HealthApp",
    "HPC",
    "OpenSSH",
    "BGL",
    "HDFS",
    "Spark",
    "Thunderbird"
]

'''"Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Mac",
    "Hadoop",
    "OpenStack",
    "HealthApp",
    "HPC",
    "OpenSSH",
    "BGL",
    "HDFS",
    "Spark",
    "Thunderbird",'''

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        },
}

def PIPLUP_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-otc', '--oracle_template_correction',
                        help="Set this if you want to use corrected oracle templates",
                        default=False, action='store_true')
    parser.add_argument('-full', '--full_data',
                        help="Set this if you want to test on full dataset",
                        default=False, action='store_true')
    parser.add_argument('--complex', type=int,
                        help="Set this if you want to test on complex dataset",
                        default=0)
    parser.add_argument('--frequent', type=int,
                        help="Set this if you want to test on frequent dataset",
                        default=0)
    # PIPLUP parameter testing
    parser.add_argument('--br_thresh', type=int,
                        help="The branch threshold for PIPLUP",
                        default=2)
    parser.add_argument('--sim_thresh', 
                        help="The similarity threshold for clustering",
                        default='default')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = PIPLUP_args()


    data_type = "full" if args.full_data else "rq1"
    input_dir = f"../../full_dataset/"
    output_dir = f"../../result/result_PIPLUP_full"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent
    )

    if args.full_data:
        datasets = datasets_full
    else:
        datasets = datasets_rq1
    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", "_full")
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        if os.path.exists(os.path.join(output_dir, f"{dataset}_full.log_structured.csv")):
            parser = None
            print("parseing result exist.")
        else:
            pass
        parser = LogParser
        # run evaluator for a dataset
        print(setting['log_format'])
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 
                'br_thresh':args.br_thresh, 'sim_thresh': args.sim_thresh,
                'keep_para':False
            },
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file
        )  # it internally saves the results into a summary file
    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"PIPLUP_{data_type}_br_thresh={args.br_thresh}_sim_thresh={args.sim_thresh}", args.complex, args.frequent)
