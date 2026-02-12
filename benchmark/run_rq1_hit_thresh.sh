# source ~/.zshrc
# conda activate logevaluate  

cd evaluation
for hit_limit in 273 385 543 666
do
    echo PIPLUP with hit_limit ${hit_limit}
    python PIPLUP_eval.py --hit_limit ${hit_limit} -otc
done
