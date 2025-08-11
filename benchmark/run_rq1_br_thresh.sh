# source ~/.zshrc
# conda activate logevaluate

cd evaluation
for br_thresh in 2 3 4 5 6
do
    echo PIPLUP with br ${br_thresh}
    python PIPLUP_eval.py --br_thresh ${br_thresh} -otc
done
