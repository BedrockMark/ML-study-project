# Part 1 Write Up

In this part I chose to go with (change model) methods, where I attempt to train the model in a different way to observe some bias from the data.

### What I did

Inspired by (Gururangan et al., 2018; Poliak et al., 2018), that pointed out: for NLI tasks, there're clues left in hypothesis, that implies the label without looking into the premise. 
Which is, we suspect that the model cheated by looking at the pattern in the hypothesis, without fully understanding and comparing to the premise that is being given. And we want the model to understand the task better.

To examine if that is true, I modified the run.py and the helper functions, so that it supports the removal of the premise section from tokenizer, and so the premise will never exist for the model. Then, I trained each of a base model, which sees the premise, and a comparision model, which does not see premise.

### Evaluate output

> Raw outputs should be included in the repo in `eval_output` folder :D

|Model | Accuracy | loss|
|-|-|-|
|Base | 89.33% | 0.2982|
|Test | 70.00% | 0.7081|

We find that, if the data was unbiased, with no artifact, the model should not be able to tell the label using hypothesis stand alone. Its behavior should be close to trivial random predictor, i.e. result an accuracy around 33.33%.

However, although we removed the premise from model, it still reached 70% accuracy. This implies some strong correlation between hypothesis sentence stand alone to the labels, that the model learnt to cheat on, and it may not learnt to understand premise fully.

This may be caused by the crowd workers are using keywords that directly connected to the label, or it's English tend to have the limited common words for each label.

### How to test

> The program is modified to detect `--no_premise` label. After adding it into the command line, the tokenizer will no longer include `examples['premise']`. So there's only the hypothesis for the model.

The following 2 represents the way I used to train and evaluate the model:

Train&evaluate base model:
```
python3 run.py --per_device_train_batch_size 72 --do_train --task nli --dataset snli --output_dir ./trained_model/
python3 run.py --do_eval --task nli --dataset stanfordnlp/snli --model ./trained_model/ --output_dir ./eval_output/  
```

Train&evaluate test model:
```
python3 run.py --per_device_train_batch_size 72 --do_train --task nli --dataset snli --output_dir ./trained_model/ --no_premise
python3 run.py --do_eval --task nli --dataset stanfordnlp/snli --model ./trained_model/ --output_dir ./eval_output/  --no_premise
```