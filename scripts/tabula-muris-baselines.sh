#!/bin/bash

# baselines
python run.py dataset=tabula_muris method=baseline n_shot=5 exp.name=baseline_tabula_muris_5_shot method.stop_epoch=40
python run.py dataset=tabula_muris method=baseline_pp n_shot=5 exp.name=baseline_pp_tabula_muris_5_shot method.stop_epoch=40

# 5-shot methods with 40 epochs
python run.py dataset=tabula_muris method=matchingnet n_shot=5 exp.name=matchingnet_tabula_muris_5_shot method.stop_epoch=40
python run.py dataset=tabula_muris method=protonet n_shot=5 exp.name=protonet_tabula_muris_5_shot method.stop_epoch=40
# MAML-specific parameters: inner LR and number of updates
python run.py dataset=tabula_muris method=maml n_shot=5 exp.name=maml_tabula_muris_5_shot_1_update method.task_update_num=1 method.maml_inner_lr=0.01 method.stop_epoch=40
python run.py dataset=tabula_muris method=maml n_shot=5 exp.name=maml_tabula_muris_5_shot_5_updates_innerlr_002 method.task_update_num=5 method.maml_inner_lr=0.002 method.stop_epoch=40

# 1-shot methods with 60 epochs
python run.py dataset=tabula_muris method=matchingnet n_shot=1 exp.name=matchingnet_tabula_muris_1_shot_60_epochs method.stop_epoch=60
python run.py dataset=tabula_muris method=protonet n_shot=1 exp.name=protonet_tabula_muris_1_shot_60_epochs method.stop_epoch=60
# MAML-specific parameters: inner LR and number of updates
python run.py dataset=tabula_muris method=maml n_shot=1 exp.name=maml_tabula_muris_5_shot_1_update_60_epochs method.task_update_num=1 method.maml_inner_lr=0.01 method.stop_epoch=60
python run.py dataset=tabula_muris method=maml n_shot=1 exp.name=maml_tabula_muris_5_shot_5_updates_innerlr_002_60_epochs method.task_update_num=5 method.maml_inner_lr=0.002 method.stop_epoch=60
