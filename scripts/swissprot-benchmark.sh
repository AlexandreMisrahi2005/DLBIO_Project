#!/bin/bash

# Training runs for Swiss Prot dataset
echo "Starting training runs for Swiss Prot dataset with 5-shot setup"

# Baseline and Baseline++
python run.py dataset=swissprot method=baseline n_shot=5 exp.name=baseline_swissprot_5_shot method.stop_epoch=40
python run.py dataset=swissprot method=baseline_pp n_shot=5 exp.name=baseline_pp_swissprot_5_shot method.stop_epoch=40

# MatchingNet
python run.py dataset=swissprot method=matchingnet n_shot=5 exp.name=matchingnet_swissprot_5_shot method.stop_epoch=40

# ProtoNet
python run.py dataset=swissprot method=protonet n_shot=5 exp.name=protonet_swissprot_5_shot method.stop_epoch=40

# MAML with various configurations
python run.py dataset=swissprot method=maml n_shot=5 exp.name=maml_swissprot_5_shot_task1_innerlr01 method.task_update_num=1 method.maml_inner_lr=0.01 method.stop_epoch=40
python run.py dataset=swissprot method=maml n_shot=5 exp.name=maml_swissprot_5_shot_task5_innerlr002 method.task_update_num=5 method.maml_inner_lr=0.002 method.stop_epoch=40


echo "All Swiss Prot 5-shot training runs are initiated"

# Training runs for Swiss Prot dataset with 1-shot setup and 60 epochs
echo "Starting training runs for Swiss Prot dataset with 1-shot setup"

# Baseline
python run.py dataset=swissprot method=baseline n_shot=1 exp.name=baseline_swissprot_1_shot method.stop_epoch=60

# Baseline++
python run.py dataset=swissprot method=baseline_pp n_shot=1 exp.name=baseline_pp_swissprot_1_shot method.stop_epoch=60

# MatchingNet
python run.py dataset=swissprot method=matchingnet n_shot=1 exp.name=matchingnet_swissprot_1_shot method.stop_epoch=60

# ProtoNet
python run.py dataset=swissprot method=protonet n_shot=1 exp.name=protonet_swissprot_1_shot method.stop_epoch=60

# MAML with task_update_num: 1 and inner_lr: 0.01
python run.py dataset=swissprot method=maml n_shot=1 exp.name=maml_swissprot_1_shot_task1_innerlr01 method.task_update_num=1 method.maml_inner_lr=0.01 method.stop_epoch=60

# MAML with task_update_num: 5 and inner_lr: 0.002
python run.py dataset=swissprot method=maml n_shot=1 exp.name=maml_swissprot_1_shot_task5_innerlr002 method.task_update_num=5 method.maml_inner_lr=0.002 method.stop_epoch=60

echo "All Swiss Prot 1-shot training runs are initiated"
