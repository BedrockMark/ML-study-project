# Fix the artifact

### What I've done
I re-implemented DRiFT trainer, which calculates the loss based on the combined logits (current model's logit + hypothesis-only's logit). By doing that, the new model learns less when hypothesis-only model can predict certain tag well, so that the model learns more when the question requires premise to be answered correctly.

Using this method, I trained the debiased model, along with a debiased model with `--no_premise` for contrast.

The results comes below:

|Model | Accuracy | loss|
|-|-|-|
|Base | 89.33% | 0.2982|
|Debiased|86.07%|0.3954|
|Hypothesis_only| 70.00% | 0.7081|
|Debiased_Hypothesis_only|21.62% | 1.1723|

We can see that, by using the new method, the model can no longer preform as well using hypothesis only. Its preformance is getting close to 33.33% (random outputs). In fact, it only 'learnt' hypothesis pattern from questions that are not so much correlated to hypothesis, this might explain why its performance is getting worse than random.

This shows that, the DRiFT training method can correctly debiase the sample, and if the model learnt the most from questions not correlated to hypothesis, then it must learnt most patterns from premise, which indicates it's debiased.

Although the debiased model has a preformance downgrade, I note it's potentially because it cheated less on hypothesis, or learnt from less samples (it learn a little in most samples, which can be solved using hypothesis only.)

### To test:

> I had the `base` model from part 1 to `.\trained_model\base_model` and `hypothesis only` model to `.\trained_model\biased_model`.

To train & evaluate a debiased new model from scratch:
```
python3 run.py --per_device_train_batch_size 128 --do_train --task nli --dataset snli --output_dir ./trained_model/ --model .\trained_model\base_model --biased_model .\trained_model\biased_model
python3 run.py --do_eval --task nli --dataset snli --model .\trained_model\debiased_model --output_dir ./eval_output/
```
To train & evaluate the contrast debiased new model with only hypothesis:
```
python3 run.py --per_device_train_batch_size 128 --do_train --task nli --dataset snli --output_dir ./trained_model/ --model .\trained_model\biased_model --biased_model .\trained_model\biased_model --no_premise
 python3 run.py --do_eval --task nli --dataset snli --model .\trained_model\debiased_hypothesis_only --output_dir ./eval_output/ --no_premise
```