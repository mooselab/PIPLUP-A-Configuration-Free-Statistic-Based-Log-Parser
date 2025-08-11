# source ~/.zshrc
# conda activate logevaluate

cd evaluation
for sim_thresh in default 0.4 0.5 0.6 0.7
do
    echo PIPLUP with sim_thresh ${sim_thresh}
    python PIPLUP_eval.py --sim_thresh ${sim_thresh} -otc
done
