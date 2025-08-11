# source ~/.zshrc
# conda activate logevaluate

cd evaluation
for technique in PIPLUP
do
    echo ${technique}
    python ${technique}_eval.py -full -otc
done
