|Model | Accuracy | loss|
|-|-|-|
|Base | 89.33% | 0.2982|
|Test | 70.00% | 0.7081|
|Debiased|86.07%|0.3954|

python3 run.py --per_device_train_batch_size 128 --do_train --task nli --dataset snli --output_dir ./trained_model/ --model .\trained_model\base_model --biased_model .\trained_model\biased_model


python3 run.py --do_eval --task nli --dataset stanfordnlp/snli --model .\trained_model\debiased_model --output_dir ./eval_output/