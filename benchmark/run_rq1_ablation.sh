# source ~/.zshrc
# conda activate logevaluate

cd evaluation
echo PIPLUP without preprocessing
python PIPLUP_eval.py -no_preprocess -otc

echo PIPLUP without merging
python PIPLUP_eval.py -no_merge -otc
