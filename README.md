# PIPLUP: Plug It and Play on Logs: configUration-free Parser
### The replication package for "Plug it and Play on Logs: A Configuration-Free Statistic-Based Log Parser". 

<img src="./figures/PIPLUP.png?raw=true" alt="piplup" width="100" height="100">

## Introduction
The overview of our tool structure is shown in the following figure. The design of PIPLUP comprises two stages: **cluster searching**, **cluster updating**. After parsing all log lines, the results are stored in a CSV file during the **result storing** stage for further verification.

![image](./figures/overview.png?raw=true)

**Cluster Searching**: Inspired by Drain, PIPLUP leverages a similar tree structure as a hashing function to find the most compatible leaf for an incoming log message and conduct further comparisons. Instead of hashing with $n$ prefixes, PIPLUP relaxes the assumption by using a tree with two fixed levels: **Constant Token Level** and Length Level. The relaxed structure can also efficiently search log clusters and match templates.

**Cluster Updating**: The matched clusters are dynamically updated to allow online parsing. First, the log line number of the incoming log message will be appended to the cluster's *log line* list, which stores a list of log indices belonging to this cluster. Then, the tokens in the message will be used to update the *message example* and the *template list*. If the *template list* is updated, the template merging process will be triggered to reduce redundancy in event templates. 

**Result Storing**: PIPLUP may create multiple templates for a log cluster. If a cluster contains only one template, then all log messages are matched with this event directly. Conversely, if multiple templates are inferred, we assign the template to in-cluster log messages using regex matching. The matched results are stored in a CSV file for further analysis.

## Dependencies
- python>=3.8
- chardet==5.1.0
- ipython==8.12.0
- matplotlib==3.7.2
- natsort==8.4.0
- numpy==1.24.4
- pandas==2.0.3
- regex==2022.3.2
- scipy
- tqdm==4.65.0
- rpy2
- spacy

## Dataset
For experiments, we leveraged the log data from **Loghub 2.0**. The original data can be obtained from the [Loghub 2.0 repository](https://github.com/logpai/loghub-2.0). Before replicating the experiments, please obtain **Loghub 2.0** from its original repository. Before starting the experiment, we corrected the event templates with the latest rules provided by [LOGPAI](https://github.com/logpai/LUNAR/blob/3f92de322030602a7c50f0e8ebceced935da199e/LUNAR/llm_module/post_process.py#L10) to ensure the quality of the ground truths. The ground truth templates can be automatically corrected with ``template_correction.py``.

## Experiments
To replicate the results in RQ1, switch folder to ``benchmark`` and run the command ``./run_rq1_br_thresh.sh`` and ``./run_rq1_sim_thresh.sh``; to replicate the overall evaluation of RQ2 and RQ3, run the command ``./run_all_full.sh``. To conduct the Scott-Knott effect size difference (ESD) analysis, please first follow the tutorial [ScottKnottESD](https://github.com/klainfo/ScottKnottESD?tab=readme-ov-file) to install the package, then run the code ``./sk_analysis.py``. 

### RQ1: How do different parameters impact PIPLUP?
Two parameters are required in PIPLUP's parsing process, namely $\theta_{sim}$ and $\theta_{br}$. $\theta_{sim}$ is a message-specific parameter used for similarity-based cluster search; $\theta_{br}$ is another globally-insensitive parameter used as a branching threshold for determining whether a token is constant. When we vary one parameter, we fix the other parameter at its default value. Our default setting for PIPLUP are: $\theta_{sim}=dynamic$ and $\theta_{br}=2$. We evaluated the impact of different settings of these parameters through a set of experiments, and the results are summarized in the following table:

![image](./figures/parameters.png?raw=true)

**Evaluating the impact of $\theta_{sim}$**: We evaluated PIPLUP's performance using multiple static $\theta_{sim}$ values (i.e., 0.4, 0.5, 0.6, and 0.7) and our message-specific, dynamic $\theta_{sim}$ to understand the impact of dynamic similarity thresholds. Our message-specific (i.e., dynamic) set $\theta_{sim}$ achieved a near-optimal performance compared to the best-performing fixed threshold, with optimal values in 5 datasets, and an average performance gap with maximum values around 1% on all datasets. The default $\theta_{sim}$ values are listed in the following table:

![image](./figures/sim_thresholds.png?raw=true)

**Evaluating the impact of $\theta_{br}$**: According to our analysis on similarity-based template clusters, the candidates for $\theta_{br}$ are 2, 3, 4, 5, and 6. Our default setting (i.e., $\theta_{br}=2$) obtained the best performance on average. Further, it also achieved the maximum values on 6 datasets and an average performance gap with the maximum value around 1\%, in terms of all four metrics. The comparison result further revealed our branching strategy's stability and applicability. 

Detailed results can be found under ``./results/RQ1/``. We inherit the parameter settings from RQ1 and use them to parse all 14 datasets in RQ2 and RQ3. 

### RQ2: How does PIPLUP compare to state-of-the-art parsers in terms of parsing effectiveness?

PIPLUP is compared with 7 state-of-the-art log parsers, including Drain, XDrain, Preprocessed-Drain, LILAC, LibreLog, LogBatcher, and LUNAR. Due to resource limitations, we did not replicate LibreLog. Therefore, the parsing performance and time consumption of this log parser are obtained from its original study. We also conducted the experiment for PILAR, which is another data-insensitive log parser. The replication code of PILAR can be found in the ``PILAR_implementation`` folder. The following table shows the parsing effectiveness of PIPLUP, along with the seven benchmarks. 

![image](./figures/effectiveness.png?raw=true)

According to the table, PIPLUP obtains significantly better average performance than state-of-the-art statistic-based parsers on all four metrics. Moreover, even with LLM-powered semantic-based parsers included, the simple PIPLUP approach is still statistically optimal or second-optimal in terms of all four metrics. It gained a similar average performance as the best unsupervised semantic-based log parser (i.e., LUNAR), indicating its high parsing performance. 

### RQ3: How does PIPLUP compare to state-of-the-art parsers in terms of parsing efficiency?

The datasets are sorted from the smallest (i.e., with the least number of lines) to the largest (i.e., with the most number of lines), and their time consumptions are documented in the following table. As shown in the following table, all parsers show anomalous high time consumption on the Thunderbird dataset. Therefore, we provided two versions of statistical time rankings (i.e., all files and files excluding Thunderbird) to avoid obtaining misleading conclusions. 

![image](./figures/efficiency.png?raw=true)

PIPLUP requires more processing time than Drain, but less time than XDrain and Preprocessed-Drain on average. It also has a much lower time consumption than the semantic-based parsers. According to the Scott-Knott ESD ranking, PIPLUP is the second most efficient. It requires only $\sim$1.5 seconds to $\sim$25 minutes to parse each of the studied datasets. Its time efficiency is statistically comparable to state-of-the-art statistic-based parsers and much better than semantic-based ones that rely on expensive computing resources. 

Detailed results for RQ2 and RQ3 can be found under ``./results/RQ2&RQ3/``. 

## Folder Structure
```
├── 2k_dataset # Loghub-2k
├── PILAR_implementation # PILAR evaluation with Loghub 2.0 evaluation functions
├── benchmark
    ├── evaluation # Configurations for the parsers
    ├── logparser # Main code for parsers
        ├── Drain
        ├── PIPLUP
        ├── Preprocessed_Drain
        ├── utils
        ├── XDrain
        └── __init__.py
    ├── old_benchmark # Default settings for the Drain series
    ├── run_all_full.sh # Script for running PIPLUP
    ├── run_rq1_br_thresh.sh
    ├── run_rq1_sim_thresh.sh
    └── README.md
├── figures 
├── result # Results for all parsers in CSV format (including PILAR)
    ├── RQ1
    └── RQ2&RQ3
├── sk_analysis.py # Code for Scott-Knott ESD analysis
├── template_correction.py # Code for ground truth template correction
└── README.md
```

