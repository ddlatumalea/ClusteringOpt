data=param_grid.csv
X=data/primates_X.csv

# create directory
mkdir -p output >/dev/null 2>&1

< $data parallel -j16 -C , --header , "python optimize.py --model TSNE --params {} --header perplexity early_exaggeration learning_rate n_iter angle method init --data $X >> results.csv"

best_kl=1000000
best_model=1000000
while IFS="," read -r model_id model_score; do
    if (( $(echo "$model_score < $best_kl" |bc -l) )); then
		best_kl=$model_score
		best_model=$model_id
	fi
done < results.csv
echo $best_model $best_kl

# list directories and delete everything except the correct one :)
find output/ -mindepth 1 -type f ! -name $best_model.pkl -exec rm -rf {} +