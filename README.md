# PIPLUP: Plug It and Play on Logs: configUration-free Parser
### The replication package for "Plug it and Play on Logs: A Configuration-Free Statistic-Based Log Parser". 

<img src="./figures/PIPLUP.png?raw=true" alt="piplup" width="100" height="100">

## Introduction
PIPLUP's parsing process is shown in the following figure. PIPLUP comprises two core parsing stages: **online log clustering**, **cluster updating**. After parsing all log lines, the results are stored in a CSV file during the **template matching** stage for further verification. 

![image](./figures/parsing_process.png?raw=true)

PIPLUP leverages a novel tree structure without assuming constant tokens' format or position, and enhances the template extraction approach based on template similarity and describability. Further, it uses a set of data-insensitive parameters, enabling users to directly "plug and play" PIPLUP on their log files without excessive configuration. 

![image](./figures/parsing_tree.png?raw=true)

**Online Log Clustering**: Inspired by Drain, PIPLUP leverages a similar tree structure as a hashing function to find the most compatible leaf for an incoming log message and conduct further comparisons. Instead of hashing with $n$ prefixes, PIPLUP relaxes the assumption by using a tree with two fixed levels: **Constant Token Level** and **Length Level**. During the log clustering stage, PIPLUP searches for the most compatible cluster for the incoming message using the key on the two dictionary levels and log similarity; if no existing cluster is found, PIPLUP creates a new cluster. 

**Cluster Updating**: The matched clusters are dynamically updated to allow online parsing. First, an in-cluster update is triggered to update the cluster (i.e., message sample, log lines, path list, and template list). If the template list is updated, an inter-cluster template-merging process among clusters under the same constant token node is triggered to reduce template redundancy. 

**Template Matching**: PIPLUP allows multiple templates to co-exist in a log cluster. If a cluster contains only one template, all log messages are directly matched to this event. Conversely, if multiple templates are inferred, PIPLUP assigns the template to in-cluster log messages using regex matching. The matched results are stored in a CSV file for further analysis.

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
For experiments, we use log data from **Loghub 2.0**. The original data can be obtained from the [Loghub 2.0 repository](https://github.com/logpai/loghub-2.0). Before replicating the experiments, please obtain **Loghub 2.0** from its original repository. Before starting the experiment, we correct the event templates with the latest rules provided by [LOGPAI](https://github.com/logpai/LUNAR/blob/3f92de322030602a7c50f0e8ebceced935da199e/LUNAR/llm_module/post_process.py#L10) to ensure the quality of the ground truths. The ground-truth templates can be automatically corrected with ``template_correction.py``.

## Experiments
To replicate the results in RQ1, switch folder to ``benchmark`` and run the command ``./run_rq1_br_thresh.sh`` and ``./run_rq1_sim_thresh.sh``; to replicate the overall evaluation of RQ2 and RQ3, run the command ``./run_all_full.sh``. To conduct the Scott-Knott effect size difference (ESD) analysis, please first follow the tutorial [ScottKnottESD](https://github.com/klainfo/ScottKnottESD?tab=readme-ov-file) to install the package, then run the code ``./sk_analysis.py``. 

### RQ1: How do different design choices impact PIPLUP?

#### Sensitive Analysis

Three parameters are required in PIPLUP's parsing process, namely $\theta_{hit}$, $\theta_{sim}$, and $\theta_{br}$. $\theta_{hit}$ is a globally-insensitive parameter used for frequency-based constant identification; $\theta_{sim}$ is a message-specific parameter used for similarity-based cluster search; $\theta_{br}$ is another globally-insensitive parameter used as a branching threshold to preserve different templates in a cluster. When we vary one parameter, we fix the other parameter at its default value. Our default setting for PIPLUP are: $\theta_{hit}=385$, $\theta_{sim}=dynamic$ and $\theta_{br}=2$. We evaluated the impact of different settings of these parameters through a set of experiments, and the results are summarized in the following table:

![image](./figures/sensitivity_analysis.png?raw=true)

**Evaluating the impact of $\theta_{hit}$**: The default $\theta_{hit}$ value (385) represents the size needed for a sample set reaching 95% confidence under 5% error margin from an unlimited population. We altered the $\theta_{hit}$ value to 273, 543, and 666. The altered values correspond to sample sizes from an unlimited population for 90%, 98%, and 99% confidence levels, under a 5% error margin. The result suggests the parameter is not sensitive to changes within a reasonable range.

**Evaluating the impact of $\theta_{sim}$**: We evaluated PIPLUP's performance using multiple static $\theta_{sim}$ values (i.e., 0.4, 0.5, 0.6, and 0.7) and our message-specific, dynamic $\theta_{sim}$ to understand the impact of dynamic similarity thresholds. Our message-specific (i.e., dynamic) set $\theta_{sim}$ achieved optimal performance, with optimal values in 5 datasets. The default $\theta_{sim}$ values are listed in the following table:

![image](./figures/sim_thresholds.png?raw=true)

**Evaluating the impact of $\theta_{br}$**: The candidates for $\theta_{br}$ are 2, 3, 4, 5, and 6. As shown in the table, when set within a reasonable range, the parameter is insensitive to changes. Our default setting (i.e., $\theta_{br}=2$) obtained the best performance on average. It achieved the maximum values on 5 datasets, with an average performance gap of less than 1% from the maximum for all four metrics. 

#### Ablation study

PIPLUP leverages a generalizable preprocessing framework to enhance parsing effectiveness and a merging process to reduce redundant templates. We conduct an ablation experiment to understand their impact on PIPLUP’s performance. The results of our ablation study are shown in the following table:

![image](./figures/ablation_study.png?raw=true)

**Evaluating the impact of log preprocessing**: PIPLUP is sensitive to disabling the preprocessing framework, reflecting a known limitation of statistic-based parsers that applies token-level constant/variable categorization (i.e., it cannot separate constants from variables coexisting in a token). Specifically, PIPLUP’s FTA and FGA decreased to less than 0.010 on the OpenStack and HealthApp datasets. The two datasets contain many messages with high volumes of combined-type tokens (i.e., tokens containing both constant and variables), such as “totalAltitude=240” and “ask=7”. PIPLUP's performance drop highlights the importance of log preprocessing in statistic-based log parsers.

**Evaluating the impact of template merging**: With the merging module disabled, PIPLUP exhibited slight reductions in its average GA (-6.2%), PA (-0.4%), FGA (-5.5%), and FTA (-2.7%). PIPLUP’s merging component can effectively reduce redundant templates: it successfully eliminated at least one overlapping template in 8 datasets, specifically 32 redundant templates in the HPC dataset. The performance degradation on Hadoop and OpenSSH is due to a single wrong merge in each dataset. The template difference comparison between PIPLUP w/ & w/o merging module can be found in ``./results/RQ1/merge_analysis``. 

Detailed results can be found under ``./results/RQ1/``. We inherit the parameter settings from RQ1 and use them to parse all 14 datasets in RQ2 and RQ3. 


### RQ2: How does PIPLUP compare to state-of-the-art parsers in terms of parsing effectiveness?

PIPLUP is compared with 7 state-of-the-art log parsers, including Drain, XDrain, Preprocessed-Drain, LILAC, LibreLog, LogBatcher, and LUNAR. Due to resource limitations, we did not replicate LibreLog. Therefore, the parsing results and time consumption of this log parser are obtained from its original study, and the evaluations are re-run on the corrected ground truths. We also experimented with PILAR, another data-insensitive log parser. PILAR's replication code is stored under the ``PILAR_implementation`` folder. The following table shows the parsing effectiveness of PIPLUP, along with seven benchmarks. 

![image](./figures/effectiveness.png?raw=true)

According to the table, PIPLUP achieved significantly higher average performance than state-of-the-art statistic-based parsers on all four metrics. Moreover, even with LLM-powered semantic-based parsers included, PIPLUP's performance is statistically optimal or near-optimal in terms of all four metrics. 

### RQ3: How does PIPLUP compare to state-of-the-art parsers in terms of parsing efficiency?

The datasets are sorted from the smallest (i.e., with the fewest lines) to the largest (i.e., with the most lines), and their time consumptions are documented in the following table. As shown in the following table, all parsers show an anomalous high time consumption on the Thunderbird dataset. Therefore, we provided two versions of statistical time rankings (i.e., all files and files excluding Thunderbird) to avoid obtaining misleading conclusions. 

![image](./figures/efficiency.png?raw=true)

On average, PIPLUP requires more processing time than Drain but less than XDrain and Preprocessed-Drain. It also has a much lower time consumption than the semantic-based parsers. According to the Scott-Knott ESD ranking, PIPLUP is the second most efficient. It requires only $\sim$1.5 seconds to $\sim$25 minutes to parse each of the studied datasets. Its time efficiency is statistically comparable to state-of-the-art statistic-based parsers and much better than semantic-based ones that rely on expensive computing resources (e.g., only using ~6\% of LUNAR's parsing time).

Theoretically, PIPLUP has the same $O(n\log m)$ algorithmic time complexity as Drain (detailed analysis in our paper). To illustrate the efficiency of PIPLUP’s major components during implementation, we collected their execution time and presented it in the following table. Among the five components, the most time-consuming are data loading (i.e., using pandas to read the log data) and preprocessing. In comparison, PIPLUP's log clustering and template extraction modules are highly effective. 

![image](./figures/module_efficiency.png?raw=true)

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
    ├── run_all_full.sh # Script for running default PIPLUP on all datasets
    ├── run_rq1_ablation.sh
    ├── run_rq1_br_thresh.sh
    ├── run_rq1_hit_thresh.sh
    ├── run_rq1_sim_thresh.sh
    └── README.md
├── figures 
├── result # Performance results for all parsers in CSV format (including PILAR)
    ├── RQ1
    ├── RQ2&RQ3
    └── result_PIPLUP_no_merge # Templates extracted by PIPLUP w/o merging module
├── sk_analysis.py # Code for Scott-Knott ESD analysis
├── template_correction.py # Code for ground truth template correction
└── README.md
```

