## Addition game

The sum game reads input from files that have an input item (summands) on each line, 
as in [this example file](data_generation_scripts/example_sum_input.txt), 
containing a list of 2 integers of size 0..4.

The game was largely adapted from [the basic game setup](play.py) and implemented in
[play_addition.py](play_addition.py), [data_readers_addition.py](data_readers_addition.py) and
[language_analysis_addition.py](language_analysis_addition.py). A data generation script was
implemented in [generate_addition.py](generate_addition.py).

To train agents on an example game:
```bash
WANDB_MODE='disabled' python -m egg.zoo.basic_games.play_addition --mode 'rl' \
  --train_data "egg/zoo/basic_games/data_generation_scripts/exp1.5-train-l300-r0.25-s42.txt" \
  --validation_data "egg/zoo/basic_games/data_generation_scripts/exp1.5-diag-l100-r0.25-s42.txt" \
  --n_max 20 --batch_size 300 --vocab_size 30 --receiver_cell "gru" --sender_cell "lstm" \
  --random_seed 42 --no_cuda --sender_embedding 80 --receiver_embedding 35 --sender_hidden 300 \
  --receiver_hidden 400 --max_len 5 --log_every 10 --n_epochs 1000 --print_validation_events \
  --save_to diag.jsonl
```
This will run the addition game on the specified training and evaluation files, print the final
validation events and save the results to a file `diag.jsonl`. 
Most of the parameters are taken from [the basic game setup](play.py). New relevant parameters are:
 - `save_to` In conjunction with `--print_validation_events` 
 will save the validation events to this file
 - `ax_hp` a json file outlining the hyper-parameters optimisation to carry out. 
 See e.g. [this file](ex1.5-ax.json).
 - `log_every` reduces the verbosity of the console logger to every this parameter runs.
 - `n_max` 1 + the highest summand appearing in training/eval data. If the highest appearing 
 summand is 9, `n_max` should be 10.

There are other new parameters that were not used, to see what they do, run
```bash
python -m egg.zoo.basic_games.play_addition --help
```

## Data generation script
To generate your own data, run 
```bash
python -m egg.zoo.basic_games.generate_addition --out $OUT --max_summand $MAX_SUMMAND 
--size $SIZE --unseen_ratio $RATIO --positional --sum_seen --seed 42
```
This generates dataset files in the `$OUT` directory in the format specified at the top, where
summands are at most `$MAX_SUMMAND -1` large. If `--size` is supplied, sub-samples `$SIZE$` examples.
If `--unseen_ratio` is supplied between 0 and 1, randomly splits the data in training and eval examples,
according to the split ratio, taking into account that for each eval example `x,y` both `x` and
`y` appear in the training set (discarding the example otherwise). if `--positional` is set, 
for an evaluation example `x,y`, `x` must be the first component and `y` the second component of
some training examples. if `--sum_seen` is set, the sum of an evaluation example 
`x+y = z` must appear in some training example `a+b=z`. `--seed` is used for reproducibility
purposes when sampling/splitting data.


## Replicating the experiments
To replicate the experiments _SmallDs_ and _LargeDs_, run the following commands:

For _SmallDs_:
```bash
WANDB_MODE='disabled' python -m egg.zoo.basic_games.play_addition --mode 'rl' \
  --train_data "egg/zoo/basic_games/data_generation_scripts/exp1.5-train-l300-r0.25-s42.txt" \
  --validation_data "egg/zoo/basic_games/data_generation_scripts/exp1.5-diag-l100-r0.25-s42.txt" \
  --n_max 20 --batch_size 300 --vocab_size 30 --receiver_cell "gru" --sender_cell "lstm" \
  --random_seed 42 --no_cuda --sender_embedding 80 --receiver_embedding 35 --sender_hidden 300 \
  --receiver_hidden 400 --max_len 5 --log_every 10 --n_epochs 1000 --print_validation_events \
  --save_to diag.jsonl
  ```

For _LargeDs_:
```bash
WANDB_MODE='disabled' python -m egg.zoo.basic_games.play_addition --mode 'rl' \
--train_data "egg/zoo/basic_games/data_generation_scripts/exp3-train-l900-r0.25-s42.txt" \
--validation_data "egg/zoo/basic_games/data_generation_scripts/exp3-diag-l300-r0.25-s42.txt" \
--n_max 40 --n_epochs 1000 --batch_size 900 --validation_batch_size 300 --max_len 5 \
--vocab_size 40 --receiver_cell "gru" --sender_cell "lstm" --random_seed 42 --num_workers 1 \
--sender_embedding 67 --receiver_embedding 93 --sender_hidden 300 --receiver_hidden 306 \
--log_every 10 --print_validation_events --save_to exp3-best-diag.jsonl
  ```

The results are visualised with WandB and can be seen 
[here](https://wandb.ai/schlevik/egg-final?workspace=user-schlevik).
