#!/bin/bash

# sh run_multitask_experiment.sh ../data/multitask_data/bert_based_novelty_task.yml ../data/multitask_data ../../data-ceph/arguana/argmining22-sharedtask/models/multitask/bert_model/novelty_no_weighting/ ./multitask bert-large-uncased_prepared_data
echo "Preparing data..."

python ../../multi-task-NLP/data_preparation.py \
  --task_file $1 \
  --data_dir $2 \
  --max_seq_len 512

echo "Start training the model..."

echo "$2$5"

python ../../multi-task-NLP/train.py \
  --data_dir "$2$5" \
  --task_file $1 \
  --learning_rate 2e-5 \
  --out_dir $3 \
  --epochs 10 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --grad_accumulation_steps 1 \
  --max_seq_len 512 \
  --log_per_updates 100 \
  --log_dir $4 \
  --limit_save 5\
  --eval_while_train\
  --test_while_train\
  --load_pretrained_classifiers